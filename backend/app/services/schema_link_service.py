# backend/app/services/schema_link_service.py
import re, gc, torch
import torch.nn.functional as F
from typing import List
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer as HFTokenizer, AutoModel as HFModel

class SchemaLinkService:
    def __init__(self, llm_path: str, emb_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # -------- stage1 LLM --------
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype="auto", device_map="auto"
        )
        self.llm_tok = AutoTokenizer.from_pretrained(llm_path)

        # -------- stage2 embedder --------
        self.emb_tok = HFTokenizer.from_pretrained(emb_path)
        self.emb = HFModel.from_pretrained(emb_path).to(self.device)
        self.emb.eval()

    # -------- utils from your script --------
    def extract_key_fields(self, text: str):
        last_key = text.rfind("The key field")
        if last_key == -1:
            last_key = text.rfind("The key fields")
            if last_key == -1:
                return []
        remaining_text = text[last_key:]
        start_bracket = remaining_text.find("[")
        end_bracket = remaining_text.find("]")
        if start_bracket == -1 or end_bracket == -1 or end_bracket < start_bracket:
            return []
        content = remaining_text[start_bracket + 1:end_bracket]
        return [item.strip() for item in content.split(",") if item.strip()]

    def extract_think_content(self, text: str) -> str:
        m = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        return m[-1].strip() if m else ""

    def extract_content_after_think(self, text: str) -> str:
        m = re.search(r"</think>(.*)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def extract_table_columns(self, schema_text: str):
        result = []
        current_table = None
        for line in schema_text.split("\n"):
            line = line.strip()
            if line.startswith("# Table:"):
                current_table = line[8:].strip()
                continue
            if current_table and line.startswith("(") and ":" in line:
                column_name = line[1:line.find(":")].strip()
                result.append(f"{current_table}.{column_name}")
        return result

    def extract_db_schema_items(self, database_text: str):
        schema_start = database_text.find("【Schema】")
        if schema_start == -1:
            return []
        schema_text = database_text[schema_start + len("【Schema】"):]
        foreign_keys_start = schema_text.find("【Foreign keys】")
        if foreign_keys_start != -1:
            schema_text = schema_text[:foreign_keys_start]
        return self.extract_table_columns(schema_text)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=128, max_length=128):
        if not texts:
            return torch.empty((0, 0), device=self.device)
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.emb_tok(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            outputs = self.emb(**inputs)
            last_hidden = outputs.last_hidden_state
            attention = inputs["attention_mask"].unsqueeze(-1)
            masked = last_hidden * attention
            summed = masked.sum(dim=1)
            counts = attention.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            emb = F.normalize(emb, p=2, dim=1)
            all_embeds.append(emb)
        return torch.cat(all_embeds, dim=0)

    # -------- stage1: predict --------
    def predict_links(self, question: str, evidence: str, database_text: str, max_new_tokens=512, retry=3) -> List[str]:
        prompt = f"""
You are a Schema Linking Expert
# Question: {question}
# Evidence: {evidence}
# Database: "{database_text}"
""".strip()

        messages = [{"role": "user", "content": prompt}]
        text = self.llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        for _ in range(retry):
            model_inputs = self.llm_tok([text], return_tensors="pt").to(self.llm.device)
            generated_ids = self.llm.generate(**model_inputs, max_new_tokens=max_new_tokens)
            generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
            response = self.llm_tok.batch_decode(generated_ids, skip_special_tokens=True)[0]

            links = self.extract_key_fields(response)
            if links:
                return links
        return []

    # -------- stage2: fix to real schema items --------
    def fix_links(self, pred_links: List[str], database_text: str) -> List[str]:
        candidates = self.extract_db_schema_items(database_text)  # real table.column list
        if not candidates or not pred_links:
            return []

        cand_emb = self.encode_texts(candidates)
        pred_emb = self.encode_texts(pred_links)

        sim = pred_emb @ cand_emb.T
        idx = torch.argmax(sim, dim=1).tolist()
        return [candidates[i] for i in idx]

def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
