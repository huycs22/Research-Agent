const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/paperqa/ui/static/app.js'), 'utf8');
const context = vm.createContext({localStorage: {getItem: () => null}});
vm.runInContext(source.slice(0, source.indexOf('function toast(')), context);

test('panel width clamping enforces responsive limits', () => {
  assert.equal(context.clamp(100, 180, 420), 180);
  assert.equal(context.clamp(300, 180, 420), 300);
  assert.equal(context.clamp(500, 180, 420), 420);
});

test('source-ID citation is one button, not nested legacy markup', () => {
  const id = 'src-0123456789abcdef01234567';
  const html = context.markdown(`Supported claim (${id}).`, [{id, name:'RAG page 2'}]);
  assert.equal((html.match(/<button /g) || []).length, 1);
  assert.ok(html.includes(`data-source-id="${id}"`));
  assert.ok(html.includes('(RAG page 2)</button>'));
  assert.ok(!html.includes('aria-label="Show source <button'));
});

test('long MAKES-QA citations use a compact label and retain the full tooltip', () => {
  const id = 'src-0123456789abcdef01234567';
  const name = 'MAKES-QA_ A multi-agent framework for knowledge graph construction and enrichment over scientific literature for question answering page 8';
  const html = context.markdown(`Supported claim (${id}).`, [{id, name, page:'8'}]);
  assert.ok(html.includes('(MAKES-QA · p. 8)</button>'));
  assert.ok(html.includes(`title="${name}"`));
  assert.ok(!html.includes(`>(${name})</button>`));
});
test('legacy citations still resolve and source labels are escaped', () => {
  assert.ok(context.markdown('Claim (Lewis2020 pages 1-2).', [{name:'Lewis2020 pages 1-2'}]).includes('citation-pill'));
  const id = 'src-0123456789abcdef01234567';
  const html = context.markdown(`<script>evil</script> (${id})`, [{id, name:'<img src=x onerror=evil>'}]);
  assert.ok(!html.includes('<script>') && !html.includes('<img'));
});

test('GitHub-style tables render as responsive semantic tables', () => {
  const id = 'src-0123456789abcdef01234567';
  const markdown = [
    '| Aspect | Self-RAG | Corrective-RAG |',
    '|:---|:---:|---:|',
    '| **Mechanism** | Reflection tokens | Evaluator (' + id + ') |',
    '| Safety | `<script>bad()</script>` | Evidence |',
  ].join('\n');
  const html = context.markdown(markdown, [{id, name:'CRAG page 2'}]);
  assert.ok(html.includes('<div class="markdown-table-wrap"'));
  assert.ok(html.includes('<table><thead><tr>'));
  assert.ok(html.includes('<th scope="col" class="align-center">Self-RAG</th>'));
  assert.ok(html.includes('<td class="align-right">Evaluator <button'));
  assert.ok(html.includes('<strong>Mechanism</strong>'));
  assert.ok(html.includes('&lt;script&gt;bad()&lt;/script&gt;'));
  assert.ok(!html.includes('| Aspect |'));
});

test('block quotes, ordered lists, fenced code, and safe links render', () => {
  const markdown = [
    '> Verification note',
    '',
    '1. First',
    '2. Second',
    '',
    '```python',
    'print("safe")',
    '```',
    '',
    '[Paper](https://example.com/paper)',
    '[Unsafe](javascript:alert(1))',
  ].join('\n');
  const html = context.markdown(markdown);
  assert.ok(html.includes('<blockquote>Verification note</blockquote>'));
  assert.ok(html.includes('<ol><li>First</li><li>Second</li></ol>'));
  assert.ok(html.includes('<pre data-language="python"><code>print(&quot;safe&quot;)</code></pre>'));
  assert.ok(html.includes('href="https://example.com/paper"'));
  assert.ok(!html.includes('href="javascript:'));
});
