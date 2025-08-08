import gradio as gr


def generate_svg(selected_node_id: str | None) -> str:
  left_nodes = [
    ("L1", 100, 60),
    ("L2", 100, 140),
    ("L3", 100, 220),
    ("L4", 100, 300),
  ]
  right_nodes = [
    ("R1", 500, 60),
    ("R2", 500, 160),
    ("R3", 500, 260),
  ]
  edges = [
    ("L1", "R1", 120, 60, 480, 60),
    ("L2", "R2", 120, 140, 480, 160),
    ("L3", "R2", 120, 220, 480, 160),
    ("L4", "R3", 120, 300, 480, 260),
  ]

  def should_show_edge(src: str, tgt: str) -> bool:
    if not selected_node_id or selected_node_id == "All":
      return True
    return (src == selected_node_id) or (tgt == selected_node_id)

  lines_markup = []
  for src, tgt, x1, y1, x2, y2 in edges:
    if should_show_edge(src, tgt):
      lines_markup.append(
        f'<line class="link" data-source="{src}" data-target="{tgt}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />'
      )

  def node_markup(node_id: str, x: int, y: int, side: str) -> str:
    cls = "left-node" if side == "left" else "right-node"
    return (
      f'<g class="node" data-node-id="{node_id}">'
      f'<circle class="{cls}" cx="{x}" cy="{y}" r="18" />'
      f'<text class="node-label" x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central">{node_id}</text>'
      f'</g>'
    )

  left_nodes_markup = [node_markup(nid, x, y, "left") for nid, x, y in left_nodes]
  right_nodes_markup = [node_markup(nid, x, y, "right") for nid, x, y in right_nodes]

  return f"""
<style>
  .viz-container {{
    width: 600px;
    height: 360px;
    margin: 0 auto;
  }}
  svg {{
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans,
                 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
  }}
  .node-label {{ font-size: 12px; fill: #111827; font-weight: 600; }}
  .left-node {{ fill: #6366f1; }}
  .right-node {{ fill: #22c55e; }}
  .link {{ stroke: #94a3b8; stroke-width: 2; }}
</style>

<div class="viz-container">
  <svg viewBox="0 0 600 360" width="600" height="360" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Simple node link demo">
    <rect width="100%" height="100%" fill="#ffffff" />

    <!-- Links -->
    {''.join(lines_markup)}

    <!-- Left nodes -->
    <g>
      {''.join(left_nodes_markup)}
    </g>

    <!-- Right nodes -->
    <g>
      {''.join(right_nodes_markup)}
    </g>
  </svg>
</div>
"""


def render(selected: str) -> str:
  if selected in (None, "", "All"):
    return generate_svg(None)
  return generate_svg(selected)


with gr.Blocks(title="Simple Node Link Demo") as demo:
  gr.Markdown(
    """
  ### Simple Node Link Demo
  Choose a node to filter the links. Click reset to show all.
  """
  )
  with gr.Row():
    # Hidden radio remains the single source of truth; it's updated by overlay buttons
    selected_radio = gr.Radio(
      choices=["All", "L1", "L2", "L3", "L4", "R1", "R2", "R3"],
      value="All",
      label="Selected node",
      interactive=True,
      visible=False,
    )

  # Main container that will hold both SVG and overlayed buttons
  with gr.Column(elem_id="viz_container"):
    html_viz = gr.HTML(value=generate_svg(None), elem_id="viz_html")
    # Invisible buttons placed inside the same container for absolute positioning
    btn_L1 = gr.Button(" ", elem_id="btn_L1", visible=True)
    btn_L2 = gr.Button(" ", elem_id="btn_L2", visible=True)
    btn_L3 = gr.Button(" ", elem_id="btn_L3", visible=True)
    btn_L4 = gr.Button(" ", elem_id="btn_L4", visible=True)
    btn_R1 = gr.Button(" ", elem_id="btn_R1", visible=True)
    btn_R2 = gr.Button(" ", elem_id="btn_R2", visible=True)
    btn_R3 = gr.Button(" ", elem_id="btn_R3", visible=True)
    btn_all = gr.Button("Reset", elem_id="btn_all", visible=True)

  # CSS to position buttons over the SVG nodes
  gr.HTML(
    """
    <style>
      #viz_container { position: relative; width: 600px; margin: 0 auto; }
      #viz_html { width: 600px; }

      #btn_L1, #btn_L2, #btn_L3, #btn_L4, #btn_R1, #btn_R2, #btn_R3 {
        position: absolute;
        width: 36px; height: 36px;
        left: 0; top: 0;
        background: transparent; border: none; padding: 0; margin: 0;
        transform: translate(-50%, -50%);
        cursor: pointer;
      }
      /* Map SVG coordinates directly: (100, 60) etc. */
      #btn_L1 { left: 100px; top: 60px; }
      #btn_L2 { left: 100px; top: 140px; }
      #btn_L3 { left: 100px; top: 220px; }
      #btn_L4 { left: 100px; top: 300px; }

      #btn_R1 { left: 500px; top: 60px; }
      #btn_R2 { left: 500px; top: 160px; }
      #btn_R3 { left: 500px; top: 260px; }

      #btn_all { position: relative; display: inline-block; margin-top: 12px; }
    </style>
    """
  )

  # Wire buttons to set the radio value then re-render
  def set_and_render(val: str):
    return val, render(val)

  btn_L1.click(lambda: set_and_render("L1"), outputs=[selected_radio, html_viz])
  btn_L2.click(lambda: set_and_render("L2"), outputs=[selected_radio, html_viz])
  btn_L3.click(lambda: set_and_render("L3"), outputs=[selected_radio, html_viz])
  btn_L4.click(lambda: set_and_render("L4"), outputs=[selected_radio, html_viz])
  btn_R1.click(lambda: set_and_render("R1"), outputs=[selected_radio, html_viz])
  btn_R2.click(lambda: set_and_render("R2"), outputs=[selected_radio, html_viz])
  btn_R3.click(lambda: set_and_render("R3"), outputs=[selected_radio, html_viz])
  btn_all.click(lambda: set_and_render("All"), outputs=[selected_radio, html_viz])

  # Also allow changing via the (hidden) radio for completeness
  selected_radio.change(render, inputs=selected_radio, outputs=html_viz)


if __name__ == "__main__":
  demo.launch()

