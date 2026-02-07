<template>
  <div v-if="!suggests?.length" class="empty">暂无可视化建议</div>

  <div v-for="(s, idx) in suggests" :key="idx" class="chart-wrap">
    <div class="title">{{ s.title || `${s.type} chart` }}</div>
    <div :ref="el => setRef(el, idx)" class="chart"></div>
  </div>
</template>

<script setup>
import * as echarts from 'echarts'
import { onBeforeUnmount, watch, nextTick } from 'vue'

const props = defineProps({
  columns: { type: Array, default: () => [] },
  rows: { type: Array, default: () => [] },
  suggests: { type: Array, default: () => [] }, // charts_suggest
})

const domRefs = []
const charts = []

function setRef(el, idx) {
  domRefs[idx] = el
}

function colIndex(name) {
  return props.columns.findIndex(c => c === name)
}

function buildOption(s) {
  const xi = colIndex(s.x)
  const yi = colIndex(s.y)
  if (xi < 0 || yi < 0) return null

  const xs = props.rows.map(r => r[xi])
  const ys = props.rows.map(r => r[yi])

  if (s.type === 'pie') {
    return {
      tooltip: { trigger: 'item' },
      series: [{
        type: 'pie',
        data: xs.map((x, i) => ({ name: String(x), value: Number(ys[i]) || 0 })),
      }],
    }
  }

  // bar / line 默认
  return {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: xs.map(x => String(x)) },
    yAxis: { type: 'value' },
    series: [{ type: s.type || 'bar', data: ys.map(v => Number(v) || 0) }],
  }
}

async function renderAll() {
  await nextTick()

  // dispose old
  charts.forEach(c => c?.dispose?.())
  charts.length = 0

  props.suggests.forEach((s, idx) => {
    const el = domRefs[idx]
    if (!el) return
    const option = buildOption(s)
    if (!option) return
    const chart = echarts.init(el)
    chart.setOption(option)
    charts[idx] = chart
  })
}

watch(
  () => [props.columns, props.rows, props.suggests],
  () => renderAll(),
  { deep: true }
)

onBeforeUnmount(() => {
  charts.forEach(c => c?.dispose?.())
})
</script>

<style scoped>
.chart-wrap { margin-top: 12px; }
.title { margin-bottom: 8px; font-weight: 600; }
.chart { width: 100%; height: 360px; }
.empty { color: #888; padding: 12px 0; }
</style>
