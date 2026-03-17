import sqlite3
import sqlparse
from sqlparse.sql import Statement, Token, TokenList
from sqlparse.tokens import Keyword, Name, Literal, Operator, Punctuation
import ast
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
from dataclasses import dataclass, field
import re
import hashlib
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


@dataclass
class SQLCandidate:
    """Data class for SQL candidate queries"""
    sql: str
    index: int
    

@dataclass
class ExecutabilityDetail:
    """Detailed executability evaluation"""
    is_executable: bool
    error_type: Optional[str]  # 'syntax', 'runtime', 'permission', None
    error_message: Optional[str]
    execution_time: float  # Execution time in milliseconds
    

@dataclass
class EvaluationScore:
    """Data class for evaluation scores"""
    executability: float  # Executability score (0 or 1)
    schema_conformity: float  # Schema conformity score (0-1)
    example_consistency: float  # Example consistency score (0-1)
    execution_time: float = 0.0  # Execution time (ms)
    complexity: float = 0.0  # SQL complexity (0-1)
    

@dataclass
class SchemaEvaluationDetail:
    """Detailed schema evaluation"""
    table_score: float
    column_score: float
    relationship_score: float
    overall_score: float
    used_tables: Set[str] = field(default_factory=set)
    used_columns: Set[str] = field(default_factory=set)
    unknown_elements: Set[str] = field(default_factory=set)
    

@dataclass
class EvaluationExplanation:
    """Evaluation result explanation"""
    score: EvaluationScore
    reasons: List[str]
    warnings: List[str]
    suggestions: List[str]


@dataclass
class SchemaElement:
    """Schema element"""
    name: str
    type: str  # 'table', 'column', 'function'
    context: Optional[str] = None  # Table name (for columns)
    alias: Optional[str] = None
    
    def __hash__(self):
        return hash((self.name, self.type, self.context))
    
    def __eq__(self, other):
        if not isinstance(other, SchemaElement):
            return False
        return (self.name == other.name and 
                self.type == other.type and 
                self.context == other.context)


class SQLFeatureExtractor:
    """SQL feature extractor for fast similarity calculation"""
    
    @staticmethod
    def extract_features(sql: str) -> np.ndarray:
        """
        Extract SQL feature vector
        
        Features include:
        - Table count
        - Column count
        - JOIN count and types
        - WHERE condition count
        - Aggregate function count
        - ORDER BY/GROUP BY presence
        - Subquery depth
        """
        features = []
        
        parsed = sqlparse.parse(sql)
        if not parsed:
            return np.zeros(20)
        
        sql_upper = sql.upper()
        
        # 1. Structural features
        features.append(sql_upper.count('SELECT'))
        features.append(sql_upper.count('FROM'))
        features.append(sql_upper.count('WHERE'))
        features.append(sql_upper.count('JOIN'))
        features.append(sql_upper.count('INNER JOIN'))
        features.append(sql_upper.count('LEFT JOIN'))
        features.append(sql_upper.count('RIGHT JOIN'))
        
        # 2. Aggregation and grouping
        features.append(sql_upper.count('GROUP BY'))
        features.append(sql_upper.count('ORDER BY'))
        features.append(sql_upper.count('HAVING'))
        
        # 3. Aggregate functions
        features.append(sql_upper.count('COUNT('))
        features.append(sql_upper.count('SUM('))
        features.append(sql_upper.count('AVG('))
        features.append(sql_upper.count('MIN('))
        features.append(sql_upper.count('MAX('))
        
        # 4. Logical operators
        features.append(sql_upper.count(' AND '))
        features.append(sql_upper.count(' OR '))
        features.append(sql_upper.count('NOT'))
        
        # 5. Other
        features.append(1 if 'DISTINCT' in sql_upper else 0)
        features.append(1 if 'LIMIT' in sql_upper else 0)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def cosine_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(features1, features2) / (norm1 * norm2))


class EnhancedSchemaExtractor:
    """Enhanced schema extractor"""
    
    def __init__(self, sql_keywords: Set[str]):
        self.sql_keywords = sql_keywords
    
    def extract_schema_elements(self, sql: str) -> Dict[str, Set[SchemaElement]]:
        """
        Extract all schema elements used in SQL
        
        Returns:
            {
                'tables': {SchemaElement, ...},
                'columns': {SchemaElement, ...},
                'functions': {SchemaElement, ...}
            }
        """
        result = {
            'tables': set(),
            'columns': set(),
            'functions': set()
        }
        
        parsed = sqlparse.parse(sql)
        if not parsed:
            return result
        
        statement = parsed[0]
        
        # Extract table names from FROM clause
        self._extract_from_tables(statement, result)
        
        # Extract column names from SELECT clause
        self._extract_select_columns(statement, result)
        
        # Extract columns from WHERE/JOIN conditions
        self._extract_condition_columns(statement, result)
        
        return result
    
    def _extract_from_tables(self, statement, result: Dict):
        """Extract table names from FROM clause"""
        from_seen = False
        
        for token in statement.tokens:
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue
            
            if from_seen:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        table_info = self._extract_table_info(identifier)
                        if table_info:
                            result['tables'].add(table_info)
                elif isinstance(token, sqlparse.sql.Identifier):
                    table_info = self._extract_table_info(token)
                    if table_info:
                        result['tables'].add(table_info)
                
                # Stop at WHERE and other keywords
                if token.ttype is Keyword and token.value.upper() in ('WHERE', 'JOIN', 'ORDER', 'GROUP', 'LIMIT'):
                    from_seen = False
    
    def _extract_select_columns(self, statement, result: Dict):
        """Extract column names from SELECT clause"""
        select_seen = False
        
        for token in statement.tokens:
            if token.ttype is Keyword and token.value.upper() == 'SELECT':
                select_seen = True
                continue
            
            if select_seen:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        col_info = self._extract_column_info(identifier)
                        if col_info:
                            result['columns'].add(col_info)
                elif isinstance(token, sqlparse.sql.Identifier):
                    col_info = self._extract_column_info(token)
                    if col_info:
                        result['columns'].add(col_info)
                
                # Stop at FROM
                if token.ttype is Keyword and token.value.upper() == 'FROM':
                    select_seen = False
    
    def _extract_condition_columns(self, statement, result: Dict):
        """Extract columns from WHERE/JOIN conditions"""
        # Simplified implementation - can be enhanced
        sql_str = str(statement)
        
        # Use regex to find potential column references
        # Pattern: word.word or standalone word (not a keyword)
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'
        matches = re.findall(pattern, sql_str)
        
        for match in matches:
            if '.' in match:
                parts = match.split('.')
                table_or_alias = parts[0].lower()
                column_name = parts[1].lower()
                
                if column_name not in self.sql_keywords:
                    result['columns'].add(SchemaElement(
                        name=column_name,
                        type='column',
                        context=table_or_alias
                    ))
            else:
                name = match.lower()
                if name not in self.sql_keywords:
                    # Could be a column
                    result['columns'].add(SchemaElement(
                        name=name,
                        type='column'
                    ))
    
    def _extract_table_info(self, identifier) -> Optional[SchemaElement]:
        """Extract table information from Identifier"""
        real_name = identifier.get_real_name()
        alias = identifier.get_alias()
        
        if real_name:
            return SchemaElement(
                name=real_name.lower(),
                type='table',
                alias=alias.lower() if alias else None
            )
        return None
    
    def _extract_column_info(self, identifier) -> Optional[SchemaElement]:
        """Extract column information from Identifier"""
        real_name = identifier.get_real_name()
        parent = identifier.get_parent_name()
        
        if real_name and real_name.lower() not in self.sql_keywords:
            return SchemaElement(
                name=real_name.lower(),
                type='column',
                context=parent.lower() if parent else None
            )
        return None


class ASTProcessor:
    """AST processor for calculating abstract syntax tree edit distance of SQL queries"""
    
    def __init__(self):
        self._ast_cache = {}
    
    def parse_sql_to_ast(self, sql: str) -> Dict:
        """Parse SQL into simplified AST representation with caching"""
        # Generate hash for caching
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        
        if sql_hash in self._ast_cache:
            return self._ast_cache[sql_hash]
        
        # Original parsing logic
        ast = self._parse_sql_to_ast_internal(sql)
        
        # Cache result
        self._ast_cache[sql_hash] = ast
        
        return ast
    
    def _parse_sql_to_ast_internal(self, sql: str) -> Dict:
        """Internal AST parsing method"""
        if not sql or not sql.strip():
            return {'type': 'Empty', 'value': '', 'tokens': []}
            
        try:
            parsed_statements = sqlparse.parse(sql)
            if not parsed_statements:
                return {'type': 'Empty', 'value': '', 'tokens': []}
                
            statement = parsed_statements[0]
            return self._build_ast_dict(statement)
        except Exception as e:
            return {'type': 'Error', 'value': str(e), 'tokens': []}
    
    def _build_ast_dict(self, token) -> Dict:
        """Recursively build AST dictionary"""
        if token is None:
            return {'type': 'None', 'value': '', 'tokens': []}
            
        token_type = type(token).__name__
        token_value = str(token).strip()
        
        if hasattr(token, 'tokens') and token.tokens:
            child_tokens = []
            for sub_token in token.tokens:
                if self._is_meaningful_token(sub_token):
                    child_ast = self._build_ast_dict(sub_token)
                    if child_ast['type'] != 'None':
                        child_tokens.append(child_ast)
            
            return {
                'type': token_type,
                'value': token_value,
                'ttype': str(token.ttype) if hasattr(token, 'ttype') and token.ttype else None,
                'tokens': child_tokens
            }
        else:
            return {
                'type': token_type,
                'value': token_value,
                'ttype': str(token.ttype) if hasattr(token, 'ttype') and token.ttype else None,
                'tokens': []
            }
    
    @staticmethod
    def _is_meaningful_token(token) -> bool:
        """Determine if token is meaningful"""
        if token is None:
            return False
            
        token_str = str(token).strip()
        if not token_str:
            return False
            
        if hasattr(token, 'ttype'):
            if token.ttype in (sqlparse.tokens.Whitespace, 
                              sqlparse.tokens.Whitespace.Newline,
                              sqlparse.tokens.Comment.Single,
                              sqlparse.tokens.Comment.Multiline):
                return False
        
        return True
    
    @staticmethod
    def _get_node_weight(node: Dict) -> int:
        """
        Calculate node weight based on importance
        
        Importance levels:
        - SELECT/FROM/WHERE/JOIN: 5 (core structure)
        - Column/table names: 3 (schema elements)
        - Operators/functions: 2 (logic)
        - Other: 1
        """
        node_type = node.get('type', '')
        ttype = node.get('ttype', '')
        value = node.get('value', '').upper()
        
        # Core SQL keywords
        core_keywords = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
                        'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'INTERSECT'}
        if any(kw in value for kw in core_keywords):
            return 5
        
        # Schema elements (Name type)
        if 'Name' in node_type or 'Identifier' in node_type:
            return 3
        
        # Operators and functions
        if 'Operator' in str(ttype) or 'Function' in node_type:
            return 2
        
        return 1
    
    @staticmethod
    def node_weight(node: Dict) -> int:
        """Calculate total node weight (recursive)"""
        if not node:
            return 0
        
        weight = ASTProcessor._get_node_weight(node)
        
        for child in node.get('tokens', []):
            weight += ASTProcessor.node_weight(child)
        
        return weight
    
    def calculate_edit_distance(self, ast1: Dict, ast2: Dict) -> float:
        """Calculate normalized edit distance between two ASTs"""
        
        def compute_edit_distance(node1: Dict, node2: Dict) -> int:
            """Compute edit distance between two AST nodes"""
            if not node1 and not node2:
                return 0
            
            if not node1:
                return self.node_weight(node2)
            if not node2:
                return self.node_weight(node1)
            
            nodes_equal = self._nodes_equal(node1, node2)
            
            tokens1 = node1.get('tokens', [])
            tokens2 = node2.get('tokens', [])
            
            if nodes_equal and not tokens1 and not tokens2:
                return 0
            
            if nodes_equal:
                return self._compute_sequence_edit_distance(tokens1, tokens2)
            else:
                substitute_cost = 1 + self._compute_sequence_edit_distance(tokens1, tokens2)
                delete_cost = self.node_weight(node1)
                insert_cost = self.node_weight(node2)
                
                return min(substitute_cost, delete_cost, insert_cost)
        
        distance = compute_edit_distance(ast1, ast2)
        
        weight1 = self.node_weight(ast1)
        weight2 = self.node_weight(ast2)
        max_weight = max(weight1, weight2)
        
        if max_weight == 0:
            return 0.0
        
        return min(1.0, distance / max_weight)
    
    @staticmethod
    def _nodes_equal(node1: Dict, node2: Dict) -> bool:
        """Determine if two AST nodes are equal"""
        if not node1 or not node2:
            return False
        
        if node1.get('type') != node2.get('type'):
            return False
        
        if node1.get('ttype') != node2.get('ttype'):
            return False
        
        critical_types = ['Keyword', 'Name', 'Literal']
        if node1.get('type') in critical_types:
            val1 = node1.get('value', '').strip().lower()
            val2 = node2.get('value', '').strip().lower()
            return val1 == val2
        
        return True
    
    def _compute_sequence_edit_distance(self, seq1: List[Dict], seq2: List[Dict]) -> int:
        """Compute edit distance between two AST node sequences"""
        m, n = len(seq1), len(seq2)
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = sum(self._node_weight_simple(seq2[k]) for k in range(j))
                elif j == 0:
                    dp[i][j] = sum(self._node_weight_simple(seq1[k]) for k in range(i))
                else:
                    node1, node2 = seq1[i-1], seq2[j-1]
                    
                    if self._nodes_equal(node1, node2):
                        substitute_cost = self._compute_sequence_edit_distance(
                            node1.get('tokens', []), node2.get('tokens', [])
                        )
                    else:
                        substitute_cost = (self._node_weight_simple(node1) + 
                                         self._node_weight_simple(node2))
                    
                    delete_cost = self._node_weight_simple(node1)
                    insert_cost = self._node_weight_simple(node2)
                    
                    dp[i][j] = min(
                        dp[i-1][j-1] + substitute_cost,
                        dp[i-1][j] + delete_cost,
                        dp[i][j-1] + insert_cost
                    )
        
        return dp[m][n]
    
    @staticmethod
    def _node_weight_simple(node: Dict) -> int:
        """Simple node weight calculation (non-recursive)"""
        return 1 if node else 0
    
    def clear_cache(self):
        """Clear AST cache"""
        self._ast_cache.clear()


class ParetoOptimal:
    """Optimized Pareto Optimal SQL Generator"""
    
    def __init__(self, database_path: str = None, max_workers: int = 4):
        """
        Initialize Pareto Optimal selector
        
        Args:
            database_path: SQLite database path for executability checking
            max_workers: Maximum number of parallel workers
        """
        self.database_path = database_path
        self.max_workers = max_workers
        self._conn = None
        self.ast_processor = ASTProcessor()
        self.feature_extractor = SQLFeatureExtractor()
        
        # Extended SQL keywords list
        self.sql_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'full', 'outer',
            'on', 'and', 'or', 'not', 'in', 'exists', 'like', 'between', 'is', 'null',
            'group', 'by', 'order', 'having', 'limit', 'offset', 'distinct', 'all',
            'union', 'intersect', 'except', 'case', 'when', 'then', 'else', 'end',
            'insert', 'update', 'delete', 'create', 'drop', 'alter', 'table', 'view',
            'index', 'into', 'values', 'set', 'as', 'asc', 'desc', 'count', 'sum',
            'avg', 'min', 'max', 'with', 'recursive', 'over', 'partition', 'window',
            'cast', 'convert', 'substring', 'trim', 'upper', 'lower', 'length',
            'coalesce', 'nullif', 'round', 'floor', 'ceil', 'abs', 'mod', 'power',
            'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'concat', 'replace'
        }
        
        self.schema_extractor = EnhancedSchemaExtractor(self.sql_keywords)
    
    def __del__(self):
        """Clean up database connection"""
        if self._conn:
            try:
                self._conn.close()
            except:
                pass
    
    def _get_connection(self):
        """Get or create database connection (connection pool)"""
        if self._conn is None and self.database_path:
            self._conn = sqlite3.connect(self.database_path)
        return self._conn
    
    def evaluate_executability(self, sql: str) -> float:
        """
        Evaluate SQL executability (simple version for backward compatibility)
        
        Args:
            sql: SQL query string
            
        Returns:
            Executability score (1.0 for executable, 0.0 for non-executable)
        """
        detail = self.evaluate_executability_detailed(sql)
        return 1.0 if detail.is_executable else 0.0
    
    def evaluate_executability_detailed(self, sql: str) -> ExecutabilityDetail:
        """
        Detailed executability evaluation
        
        Returns:
            ExecutabilityDetail with error type, message, and execution time
        """
        if not self.database_path:
            # Basic syntax check
            try:
                parsed = sqlparse.parse(sql)
                if not parsed:
                    return ExecutabilityDetail(False, 'syntax', 'Invalid SQL', 0)
                return ExecutabilityDetail(True, None, None, 0)
            except Exception as e:
                return ExecutabilityDetail(False, 'syntax', str(e), 0)
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            start = time.time()
            limited_sql = self._add_limit_to_sql(sql, 1)
            cursor.execute(limited_sql)
            cursor.fetchall()
            exec_time = (time.time() - start) * 1000  # Convert to milliseconds
            
            return ExecutabilityDetail(True, None, None, exec_time)
            
        except sqlite3.OperationalError as e:
            return ExecutabilityDetail(False, 'syntax', str(e), 0)
        except sqlite3.DatabaseError as e:
            return ExecutabilityDetail(False, 'runtime', str(e), 0)
        except Exception as e:
            return ExecutabilityDetail(False, 'unknown', str(e), 0)
    
    def _add_limit_to_sql(self, sql: str, limit: int) -> str:
        """Smart LIMIT clause addition"""
        sql = sql.strip().rstrip(';')
        
        # Check if LIMIT already exists (case-insensitive)
        sql_upper = sql.upper()
        
        # Use regex to detect LIMIT keyword (not in strings)
        if re.search(r'\bLIMIT\b', sql_upper):
            return sql
        
        return f"{sql} LIMIT {limit}"
    
    def evaluate_schema_conformity(self, sql: str, schema_links: Set[str]) -> float:
        """
        Evaluate schema conformity (simple version for backward compatibility)
        
        Args:
            sql: SQL query string
            schema_links: Set of schema links (table and column names)
            
        Returns:
            Schema conformity score (between 0-1)
        """
        schema_used = self._extract_schema_from_sql(sql)
        
        if not schema_used and not schema_links:
            return 1.0
        
        if not schema_used:
            return 0.0
            
        if not schema_links:
            return 0.0
        
        intersection = schema_used.intersection(schema_links)
        union = schema_used.union(schema_links)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        coverage = len(intersection) / len(schema_used) if schema_used else 0.0
        
        return (jaccard_similarity + coverage) / 2.0
    
    def evaluate_schema_conformity_detailed(
        self, 
        sql: str, 
        schema_links: Set[str]
    ) -> SchemaEvaluationDetail:
        """
        Detailed schema conformity evaluation
        
        Returns:
            SchemaEvaluationDetail with separate scores for tables, columns, etc.
        """
        schema_used = self._extract_schema_from_sql(sql)
        
        if not schema_used and not schema_links:
            return SchemaEvaluationDetail(1.0, 1.0, 1.0, 1.0)
        
        if not schema_used or not schema_links:
            return SchemaEvaluationDetail(0.0, 0.0, 0.0, 0.0)
        
        # Calculate metrics
        intersection = schema_used.intersection(schema_links)
        unknown = schema_used - schema_links
        
        # Precision and recall
        precision = len(intersection) / len(schema_used) if schema_used else 0.0
        recall = len(intersection) / len(schema_links) if schema_links else 0.0
        
        # F1 score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        return SchemaEvaluationDetail(
            table_score=f1_score,  # Simplified - same score for all
            column_score=f1_score,
            relationship_score=1.0,  # Simplified
            overall_score=f1_score,
            used_tables=schema_used,
            used_columns=schema_used,
            unknown_elements=unknown
        )
    
    def _extract_schema_from_sql(self, sql: str) -> Set[str]:
        """Extract schema elements (table and column names) used in SQL"""
        schema_elements = set()
        
        sql_cleaned = self._remove_string_literals(sql)
        
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', sql_cleaned)
        
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.sql_keywords:
                schema_elements.add(word_lower)
        
        dot_patterns = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_]*)\b', sql_cleaned)
        for pattern in dot_patterns:
            parts = pattern.split('.')
            for part in parts:
                part_lower = part.lower()
                if part_lower not in self.sql_keywords:
                    schema_elements.add(part_lower)
        
        return schema_elements
    
    def _remove_string_literals(self, sql: str) -> str:
        """Remove string literals from SQL"""
        sql = re.sub(r"'[^']*'", "''", sql)
        sql = re.sub(r'"[^"]*"', '""', sql)
        sql = re.sub(r'`[^`]*`', '``', sql)
        return sql
    
    def calculate_sql_complexity(self, sql: str) -> float:
        """
        Calculate SQL complexity
        
        Considers:
        - JOIN count
        - Subquery depth
        - WHERE condition complexity
        - Aggregate function count
        """
        sql_upper = sql.upper()
        
        complexity_score = 0.0
        
        # JOIN complexity
        complexity_score += sql_upper.count('JOIN') * 2
        
        # Subquery complexity
        complexity_score += (sql_upper.count('SELECT') - 1) * 3  # -1 for main SELECT
        
        # WHERE condition complexity
        complexity_score += sql_upper.count('AND') * 0.5
        complexity_score += sql_upper.count('OR') * 0.5
        
        # Aggregation and grouping
        complexity_score += sql_upper.count('GROUP BY') * 1.5
        complexity_score += sql_upper.count('HAVING') * 1.5
        
        # Normalize to 0-1
        return min(1.0, complexity_score / 20.0)
    
    def evaluate_example_consistency(self, sql: str, examples: List[str]) -> float:
        """
        Evaluate example consistency (hybrid approach)
        
        Args:
            sql: Candidate SQL query
            examples: List of example SQL queries
            
        Returns:
            Example consistency score (between 0-1)
        """
        if not examples:
            return 0.0
        
        return self.evaluate_example_consistency_hybrid(sql, examples)
    
    def evaluate_example_consistency_hybrid(
        self, 
        sql: str, 
        examples: List[str],
        feature_threshold: float = 0.5
    ) -> float:
        """
        Hybrid evaluation: combine AST edit distance and feature vector similarity
        
        Args:
            sql: Candidate SQL
            examples: Example SQLs
            feature_threshold: Threshold for using AST comparison
        """
        if not examples:
            return 0.0
        
        # 1. Fast filtering: use feature vectors
        sql_features = self.feature_extractor.extract_features(sql)
        example_features = [self.feature_extractor.extract_features(ex) for ex in examples]
        
        feature_similarities = [
            self.feature_extractor.cosine_similarity(sql_features, ex_feat)
            for ex_feat in example_features
        ]
        
        # 2. For high-similarity examples, use precise AST comparison
        final_similarities = []
        for i, feat_sim in enumerate(feature_similarities):
            if feat_sim > feature_threshold:
                # Use AST for precise comparison
                sql_ast = self.ast_processor.parse_sql_to_ast(sql)
                example_ast = self.ast_processor.parse_sql_to_ast(examples[i])
                distance = self.ast_processor.calculate_edit_distance(sql_ast, example_ast)
                ast_sim = 1.0 - distance
                
                # Combine both similarities
                final_sim = 0.3 * feat_sim + 0.7 * ast_sim
            else:
                # Low similarity, use feature similarity directly
                final_sim = feat_sim
            
            final_similarities.append(max(0.0, final_sim))
        
        return sum(final_similarities) / len(final_similarities)
    
    def prefilter_candidates(
        self,
        candidates: List[str],
        schema_links: Set[str],
        min_schema_score: float = 0.3
    ) -> List[str]:
        """
        Pre-filter candidate SQLs
        
        Quickly remove obviously unqualified candidates
        """
        filtered = []
        
        for sql in candidates:
            # 1. Basic syntax check
            try:
                parsed = sqlparse.parse(sql)
                if not parsed:
                    continue
            except:
                continue
            
            # 2. Quick schema check
            schema_used = self._extract_schema_from_sql(sql)
            if schema_used:
                overlap = len(schema_used & schema_links)
                if overlap / len(schema_used) < min_schema_score:
                    continue
            
            # 3. SQL length reasonability
            if len(sql) < 10 or len(sql) > 5000:
                continue
            
            filtered.append(sql)
        
        return filtered
    
    def evaluate_sql_candidates(
        self, 
        candidates: List[str], 
        schema_links: Set[str], 
        examples: List[str]
    ) -> List[Tuple[SQLCandidate, EvaluationScore]]:
        """
        Evaluate all SQL candidate queries
        
        Args:
            candidates: List of SQL candidate queries
            schema_links: Schema link information
            examples: List of example SQL queries
            
        Returns:
            List of candidate queries and their evaluation scores
        """
        evaluated_candidates = []
        
        for i, sql in enumerate(candidates):
            candidate = SQLCandidate(sql=sql, index=i)
            
            # Evaluate three dimensions
            exec_detail = self.evaluate_executability_detailed(sql)
            executability = 1.0 if exec_detail.is_executable else 0.0
            schema_conformity = self.evaluate_schema_conformity(sql, schema_links)
            example_consistency = self.evaluate_example_consistency(sql, examples)
            complexity = self.calculate_sql_complexity(sql)
            
            score = EvaluationScore(
                executability=executability,
                schema_conformity=schema_conformity,
                example_consistency=example_consistency,
                execution_time=exec_detail.execution_time,
                complexity=complexity
            )
            
            evaluated_candidates.append((candidate, score))
        
        return evaluated_candidates
    
    def evaluate_sql_candidates_parallel(
        self,
        candidates: List[str],
        schema_links: Set[str],
        examples: List[str]
    ) -> List[Tuple[SQLCandidate, EvaluationScore]]:
        """
        Parallel evaluation of SQL candidates
        """
        evaluated_candidates = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_index = {
                executor.submit(
                    self._evaluate_single_candidate,
                    sql, i, schema_links, examples
                ): i
                for i, sql in enumerate(candidates)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    evaluated_candidates.append(result)
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Error evaluating candidate {index}: {e}")
        
        # Sort by original index
        evaluated_candidates.sort(key=lambda x: x[0].index)
        
        return evaluated_candidates
    
    def _evaluate_single_candidate(
        self,
        sql: str,
        index: int,
        schema_links: Set[str],
        examples: List[str]
    ) -> Tuple[SQLCandidate, EvaluationScore]:
        """Evaluate single candidate"""
        candidate = SQLCandidate(sql=sql, index=index)
        
        exec_detail = self.evaluate_executability_detailed(sql)
        executability = 1.0 if exec_detail.is_executable else 0.0
        schema_conformity = self.evaluate_schema_conformity(sql, schema_links)
        example_consistency = self.evaluate_example_consistency(sql, examples)
        complexity = self.calculate_sql_complexity(sql)
        
        score = EvaluationScore(
            executability=executability,
            schema_conformity=schema_conformity,
            example_consistency=example_consistency,
            execution_time=exec_detail.execution_time,
            complexity=complexity
        )
        
        return (candidate, score)
    
    def find_pareto_optimal(
        self, 
        evaluated_candidates: List[Tuple[SQLCandidate, EvaluationScore]]
    ) -> List[SQLCandidate]:
        """
        Find Pareto optimal solution set
        
        Args:
            evaluated_candidates: List of evaluated candidate queries
            
        Returns:
            List of Pareto optimal SQL candidate queries
        """
        # Filter out non-executable queries
        executable_candidates = [
            (candidate, score) for candidate, score in evaluated_candidates
            if score.executability > 0.0
        ]
        
        if not executable_candidates:
            return []
        
        return self.find_pareto_optimal_efficient(executable_candidates)
    
    def find_pareto_optimal_efficient(
        self, 
        evaluated_candidates: List[Tuple[SQLCandidate, EvaluationScore]]
    ) -> List[SQLCandidate]:
        """
        Efficient Pareto front finding algorithm
        
        Time complexity: O(n²) in worst case, but faster in practice
        """
        pareto_optimal = []
        
        for i, (candidate_i, score_i) in enumerate(evaluated_candidates):
            is_dominated = False
            
            for j, (candidate_j, score_j) in enumerate(evaluated_candidates):
                if i == j:
                    continue
                
                # Check if candidate_i is dominated by candidate_j
                # Dominated: all objectives >= and at least one >
                if (score_j.schema_conformity >= score_i.schema_conformity and
                    score_j.example_consistency >= score_i.example_consistency and
                    (score_j.schema_conformity > score_i.schema_conformity or
                     score_j.example_consistency > score_i.example_consistency)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(candidate_i)
        
        return pareto_optimal
    
    def explain_evaluation(
        self,
        sql: str,
        score: EvaluationScore,
        schema_links: Set[str],
        examples: List[str]
    ) -> EvaluationExplanation:
        """
        Generate explainable evaluation results
        """
        reasons = []
        warnings = []
        suggestions = []
        
        # Executability explanation
        if score.executability == 0:
            reasons.append("SQL cannot be executed, possibly due to syntax errors or schema mismatch")
            suggestions.append("Check SQL syntax and table/column names used")
        else:
            reasons.append("SQL can execute successfully")
            if score.execution_time > 100:
                warnings.append(f"Execution time is relatively long ({score.execution_time:.1f}ms)")
        
        # Schema conformity explanation
        schema_detail = self.evaluate_schema_conformity_detailed(sql, schema_links)
        
        if schema_detail.unknown_elements:
            warnings.append(f"Unknown schema elements used: {schema_detail.unknown_elements}")
            suggestions.append("Verify if these table/column names are correct")
        
        if score.schema_conformity < 0.5:
            reasons.append(f"Low schema conformity ({score.schema_conformity:.2f})")
            suggestions.append("Consider using more correct table and column names")
        elif score.schema_conformity > 0.8:
            reasons.append(f"High schema conformity ({score.schema_conformity:.2f})")
        
        # Example consistency explanation
        if score.example_consistency < 0.3:
            reasons.append("Structure differs significantly from example SQLs")
            suggestions.append("Refer to the structure and patterns in example SQLs")
        elif score.example_consistency > 0.7:
            reasons.append("Structure is very similar to example SQLs")
        
        # Complexity explanation
        if score.complexity > 0.7:
            warnings.append(f"SQL is relatively complex (complexity: {score.complexity:.2f})")
            suggestions.append("Consider simplifying the query if possible")
        
        return EvaluationExplanation(
            score=score,
            reasons=reasons,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def select_final_sql(
        self, 
        candidates: List[str], 
        schema_links: Set[str], 
        examples: List[str],
        selection_strategy: str = "balanced",
        use_parallel: bool = False,
        use_prefilter: bool = True
    ) -> str:
        """
        Select the final SQL query
        
        Args:
            candidates: List of SQL candidate queries
            schema_links: Schema link information
            examples: List of example SQL queries
            selection_strategy: Selection strategy ("balanced", "schema_priority", "example_priority")
            use_parallel: Whether to use parallel evaluation
            use_prefilter: Whether to pre-filter candidates
            
        Returns:
            The selected final SQL query
        """
        if not candidates:
            return ""
        
        # Pre-filter candidates if enabled
        if use_prefilter:
            candidates = self.prefilter_candidates(candidates, schema_links)
            if not candidates:
                return ""
        
        # Evaluate all candidate queries
        if use_parallel and len(candidates) > 10:
            evaluated_candidates = self.evaluate_sql_candidates_parallel(
                candidates, schema_links, examples
            )
        else:
            evaluated_candidates = self.evaluate_sql_candidates(
                candidates, schema_links, examples
            )
        
        # Find Pareto optimal solutions
        pareto_optimal = self.find_pareto_optimal(evaluated_candidates)
        
        if not pareto_optimal:
            # If no Pareto optimal solutions, return first executable query
            for candidate, score in evaluated_candidates:
                if score.executability > 0.0:
                    return candidate.sql
            return candidates[0]
        
        if len(pareto_optimal) == 1:
            return pareto_optimal[0].sql
        
        # Select based on strategy
        best_candidate = None
        best_score = -1.0
        
        for candidate in pareto_optimal:
            # Get corresponding evaluation score
            score = None
            for c, s in evaluated_candidates:
                if c.index == candidate.index:
                    score = s
                    break
            
            if score is None:
                continue
            
            # Calculate combined score based on selection strategy
            if selection_strategy == "balanced":
                combined_score = (score.schema_conformity + score.example_consistency) / 2.0
            elif selection_strategy == "schema_priority":
                combined_score = 0.7 * score.schema_conformity + 0.3 * score.example_consistency
            elif selection_strategy == "example_priority":
                combined_score = 0.3 * score.schema_conformity + 0.7 * score.example_consistency
            else:
                combined_score = (score.schema_conformity + score.example_consistency) / 2.0
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        return best_candidate.sql if best_candidate else pareto_optimal[0].sql


# Benchmark and testing utilities
def benchmark_evaluation_method(
    method,
    candidates: List[str],
    schema_links: Set[str],
    examples: List[str],
    runs: int = 10
) -> Dict[str, float]:
    """
    Performance benchmark for evaluation methods
    """
    times = []
    
    for _ in range(runs):
        start = time.time()
        result = method(candidates, schema_links, examples)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times))
    }


def demo_pareto_optimal_selection():
    """Demonstrate the usage of Pareto optimal selection"""
    
    # Create PO instance
    po = ParetoOptimal()
    
    # Example data
    candidate_sqls = [
        "SELECT name FROM customers WHERE age > 25",
        "SELECT customer_name FROM customer WHERE customer_age > 25",
        "SELECT c.name FROM customers c WHERE c.age > 25 ORDER BY c.name",
        "SELECT * FROM customers WHERE age > 25",
        "SELECT name, age FROM customers WHERE age > 25 AND city = 'New York'"
    ]
    
    schema_links = {"customers", "name", "age", "city", "customer_id"}
    
    example_sqls = [
        "SELECT name FROM employees WHERE salary > 50000",
        "SELECT product_name FROM products WHERE price > 100"
    ]
    
    print("=== Optimized SQL Evaluation System ===\n")
    
    print("=== Pre-filtering Test ===")
    filtered = po.prefilter_candidates(candidate_sqls, schema_links)
    print(f"Original candidates: {len(candidate_sqls)}")
    print(f"After filtering: {len(filtered)}")
    print()
    
    print("=== Feature Extraction Test ===")
    for i, sql in enumerate(candidate_sqls[:3]):
        features = po.feature_extractor.extract_features(sql)
        print(f"SQL {i+1}: {sql}")
        print(f"Feature vector shape: {features.shape}")
        print(f"Non-zero features: {np.count_nonzero(features)}")
        print()
    
    print("=== Schema Extraction Test ===")
    for i, sql in enumerate(candidate_sqls[:3]):
        extracted = po._extract_schema_from_sql(sql)
        print(f"SQL {i+1}: {sql}")
        print(f"Extracted schema: {extracted}")
        
        # Detailed evaluation
        detail = po.evaluate_schema_conformity_detailed(sql, schema_links)
        print(f"Overall score: {detail.overall_score:.3f}")
        print(f"Unknown elements: {detail.unknown_elements}")
        print()
    
    # Select final SQL
    print("=== Final SQL Selection ===")
    final_sql = po.select_final_sql(
        candidates=candidate_sqls,
        schema_links=schema_links,
        examples=example_sqls,
        selection_strategy="balanced",
        use_parallel=False,
        use_prefilter=True
    )
    
    print("Candidate SQL queries:")
    for i, sql in enumerate(candidate_sqls):
        print(f"{i+1}. {sql}")
    
    print(f"\nSelected final SQL: {final_sql}")
    
    # Show detailed evaluation information
    print("\n=== Detailed Evaluation Results ===")
    evaluated = po.evaluate_sql_candidates(candidate_sqls, schema_links, example_sqls)
    
    for candidate, score in evaluated:
        print(f"\nSQL {candidate.index + 1}:")
        print(f"  Executability: {score.executability:.3f}")
        print(f"  Schema conformity: {score.schema_conformity:.3f}")
        print(f"  Example consistency: {score.example_consistency:.3f}")
        print(f"  Complexity: {score.complexity:.3f}")
        print(f"  Execution time: {score.execution_time:.3f}ms")
        
        # Generate explanation
        explanation = po.explain_evaluation(
            candidate.sql, score, schema_links, example_sqls
        )
        
        if explanation.reasons:
            print(f"  Reasons: {'; '.join(explanation.reasons)}")
        if explanation.warnings:
            print(f"  Warnings: {'; '.join(explanation.warnings)}")
        if explanation.suggestions:
            print(f"  Suggestions: {'; '.join(explanation.suggestions)}")
    
    # Show Pareto optimal solutions
    pareto_optimal = po.find_pareto_optimal(evaluated)
    print("\n=== Pareto Optimal Solutions ===")
    for candidate in pareto_optimal:
        print(f"  SQL {candidate.index + 1}: {candidate.sql}")
    
    # Performance comparison
    print("\n=== Performance Benchmark ===")
    if len(candidate_sqls) >= 5:
        print("Testing sequential evaluation...")
        seq_perf = benchmark_evaluation_method(
            lambda c, s, e: po.evaluate_sql_candidates(c, s, e),
            candidate_sqls, schema_links, example_sqls,
            runs=5
        )
        
        print("Testing parallel evaluation...")
        par_perf = benchmark_evaluation_method(
            lambda c, s, e: po.evaluate_sql_candidates_parallel(c, s, e),
            candidate_sqls, schema_links, example_sqls,
            runs=5
        )
        
        print(f"Sequential - Mean: {seq_perf['mean_time']:.4f}s, Std: {seq_perf['std_time']:.4f}s")
        print(f"Parallel   - Mean: {par_perf['mean_time']:.4f}s, Std: {par_perf['std_time']:.4f}s")
        
        if par_perf['mean_time'] > 0:
            speedup = seq_perf['mean_time'] / par_perf['mean_time']
            print(f"Speedup: {speedup:.2f}x")
    
    # Clear cache
    po.ast_processor.clear_cache()
    print("\nAST cache cleared.")


if __name__ == "__main__":
    demo_pareto_optimal_selection()