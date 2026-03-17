<template>
  <div v-if="emptyReason" class="empty">{{ emptyReason }}</div>

  <div v-else>
    <div v-for="(s, idx) in suggests" :key="idx" class="chart-wrap">
      <div class="title">{{ s.title || defaultTitle(s) }}</div>
      <div :ref="el => setRef(el, idx)" class="chart"></div>
    </div>
  </div>
</template>

<script setup>
import * as echarts from 'echarts'
import { computed, onBeforeUnmount, onMounted, watch, nextTick } from 'vue'

const props = defineProps({
  columns: { type: Array, default: () => [] },
  rows: { type: Array, default: () => [] },
  suggests: { type: Array, default: () => [] }, // charts_suggest
})

/** 调色板：ECharts 会按数据项自动轮换颜色 */
const COLORS = [
  '#5470C6', '#91CC75', '#EE6666', '#FAC858', '#73C0DE',
  '#3BA272', '#FC8452', '#9A60B4', '#EA7CCC',
]

const domRefs = []
const charts = []

function setRef(el, idx) {
  domRefs[idx] = el
}

function defaultTitle(s) {
  const t = (s?.type || 'chart').toUpperCase()
  const x = pickFieldName(s, 'x') || pickFieldName(s, 'name') || ''
  const y = pickFieldName(s, 'y') || pickFieldName(s, 'value') || ''
  return x && y ? `${t}: ${x} vs ${y}` : `${t} chart`
}

function colIndex(name) {
  return props.columns.findIndex(c => c === name)
}

/** 兼容后端两种结构：x/y 或 name/value；字段可能是 string 或 {field: "..."} */
function pickFieldName(s, key) {
  const v = s?.[key]
  if (!v) return undefined
  if (typeof v === 'string') return v
  return v?.field
}

function buildSeriesDataBar(ys) {
  // 让每根柱子/每个点自动多色：把 value 包成对象也行，但其实设置 color 就足够了
  return ys.map(v => Number(v) || 0)
}

function buildOption(s) {
  const type = s?.type || 'bar'

  // 兼容：pie 用 name/value；bar/line 用 x/y
  const xName = pickFieldName(s, 'x') ?? pickFieldName(s, 'name')
  const yName = pickFieldName(s, 'y') ?? pickFieldName(s, 'value')

  if (!xName || !yName) {
    console.warn('[ChartPanel] missing x/y field in suggest:', s)
    return null
  }

  const xi = colIndex(xName)
  const yi = colIndex(yName)
  if (xi < 0 || yi < 0) {
    console.warn('[ChartPanel] field not found in columns:', { xName, yName, columns: props.columns })
    return null
  }

  const xs = props.rows.map(r => r?.[xi])
  const ys = props.rows.map(r => r?.[yi])

  // 通用基础配置（含多色）
  const base = {
    color: COLORS,
    animation: true,
    tooltip: { confine: true },
  }

  if (type === 'pie') {
    // 取列
    const xi0 = xi
    const yi0 = yi

    // 抽样判断类型：Year 应该更像类别(字符串/年份)，proportion 更像数值
    const sampleX = props.rows[0]?.[xi0]
    const sampleY = props.rows[0]?.[yi0]

    const xLooksNumeric = sampleX != null && sampleX !== '' && !Number.isNaN(Number(sampleX))
    const yLooksNumeric = sampleY != null && sampleY !== '' && !Number.isNaN(Number(sampleY))

    // 如果“name 是数值而 value 不是数值”，大概率反了：交换
    let nameIdx = xi0
    let valueIdx = yi0
    if (xLooksNumeric && !yLooksNumeric) {
      nameIdx = yi0
      valueIdx = xi0
    }

    const data = props.rows.map(r => ({
      name: String(r?.[nameIdx] ?? ''),
      value: Number(r?.[valueIdx]) || 0,
    }))

    return {
      ...base,
      tooltip: { ...base.tooltip, trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { type: 'scroll', orient: 'vertical', left: 10, top: 10, bottom: 10 },
      series: [{
        type: 'pie',
        radius: ['35%', '70%'],
        center: ['60%', '50%'],
        avoidLabelOverlap: true,
        label: { show: true, formatter: '{b}: {c}' },
        data,
      }],
    }
  }


    const xData = xs.map(x => String(x ?? ''))
    const yData = ys.map((v, i) => ({
      value: Number(v) || 0,
      itemStyle: { color: COLORS[i % COLORS.length] },
    }))

    // 数据点很多时更好用
    const needZoom = xData.length > 12

    return {
      ...base,
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: { left: 48, right: 20, top: 20, bottom: needZoom ? 60 : 40, containLabel: true },
      xAxis: {
        type: 'category',
        data: xData,
        axisLabel: { interval: 0, rotate: xData.length > 8 ? 30 : 0 },
      },
      yAxis: { type: 'value' },
      dataZoom: needZoom
        ? [{ type: 'slider', bottom: 10, height: 18 }, { type: 'inside' }]
        : [],
      series: [{
        type: type === 'line' ? 'line' : 'bar',
        data: yData,
        smooth: type === 'line',
        // line 更好看点
        symbol: type === 'line' ? 'circle' : 'emptyCircle',
        symbolSize: type === 'line' ? 6 : 4,
      }],
    }
  }

const emptyReason = computed(() => {
  if (!props.suggests?.length) return '暂无可视化建议'
  // 有 suggests 但可能都 buildOption 失败
  const anyOk = props.suggests.some(s => buildOption(s))
  if (!anyOk) return '可视化建议无法匹配当前结果列（x/y 字段不在 columns 中）'
  return ''
})

function getOrInitChart(idx, el) {
  const existing = charts[idx]
  if (existing && !existing.isDisposed?.()) return existing
  const chart = echarts.init(el)
  charts[idx] = chart
  return chart
}

async function renderAll() {
  await nextTick()

  props.suggests.forEach((s, idx) => {
    const el = domRefs[idx]
    if (!el) return

    const option = buildOption(s)
    if (!option) return

    const chart = getOrInitChart(idx, el)
    chart.setOption(option, { notMerge: true, lazyUpdate: true })
    chart.resize()
  })
}

function resizeAll() {
  charts.forEach(c => c?.resize?.())
}

let ro
onMounted(() => {
  window.addEventListener('resize', resizeAll)
  // 容器尺寸变化也 resize（比如侧边栏收起/展开）
  ro = new ResizeObserver(() => resizeAll())
  domRefs.forEach(el => el && ro.observe(el))
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeAll)
  ro?.disconnect?.()
  charts.forEach(c => c?.dispose?.())
})

watch(
  () => [props.columns, props.rows, props.suggests],
  () => renderAll(),
  { deep: true, immediate: true }
)
</script>

<style scoped>
.chart-wrap { margin-top: 12px; }
.title { margin-bottom: 8px; font-weight: 600; }
.chart { width: 100%; height: 360px; }
.empty { color: #888; padding: 12px 0; }
</style>
