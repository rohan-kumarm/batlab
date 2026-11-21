"""Builds a local marking page: you click body points on release frames to create true values."""

# base64 turns JPEG bytes into text that can sit inside the HTML file.
import base64
# json writes the page's data block.
import json
# Path builds output paths.
from pathlib import Path

# OpenCV encodes frames as JPEG.
import cv2

# Region around the batter, worked out from the stumps.
from .calibrate import Calibration, batter_roi
# Frame reading.
from .video import VideoInfo, read_frames

# The points to mark, in order: (key saved in labels.json, instruction shown on the page).
# Eyes and feet are named by picture side, not body side, so marking never depends on
# knowing which leg or eye is which.
POINTS = [
    ("eye_left_in_picture", "Eye on the LEFT side of the picture"),
    ("eye_right_in_picture", "Eye on the RIGHT side of the picture"),
    ("hands", "Centre of your hands on the bat handle"),
    ("front_knee", "Centre of your front knee (the pad nearest the stumps)"),
    ("feet_left_edge", "Left-most point of your shoes"),
    ("feet_right_edge", "Right-most point of your shoes"),
]
# Frames to mark per delivery, as frame counts before release (12 and 6 frames = 0.2 s and 0.1 s
# at 60 fps). The two earlier frames give a true value for head stillness.
FRAMES_BEFORE = (12, 6, 0)


def build_label_page(info: VideoInfo, cal: Calibration, release_frames: list[int], out: Path) -> Path:
    """Write index.html with every frame to mark embedded in it; return its path."""
    # Make the output folder.
    out.mkdir(parents=True, exist_ok=True)
    # Region around the batter, clamped to the frame.
    x1, y1, x2, y2 = batter_roi(cal)
    # Clamp the left and top edges.
    x1, y1 = max(0, x1), max(0, y1)
    # Clamp the right and bottom edges.
    x2, y2 = min(info.width, x2), min(info.height, y2)
    # One entry per frame to mark.
    frames = []
    # Go through deliveries in order (numbered from 1).
    for d, rf in enumerate(release_frames, start=1):
        # Each of the frames before and at release.
        for back in FRAMES_BEFORE:
            # Frame number to read.
            n = rf - back
            # Read that single frame at full resolution.
            img = read_frames(info, n / info.fps, 1)[0]
            # Cut out the batter region and encode it as a good-quality JPEG.
            ok, jpg = cv2.imencode(".jpg", img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])
            # Store what the page and the evaluator need.
            frames.append({
                # Delivery number.
                "delivery": d,
                # Frame number in the video.
                "frame": n,
                # How many frames before release (0 = the release frame).
                "frames_before_release": back,
                # Where the crop sits in the full frame, to turn clicks into full-frame pixels.
                "offset": [x1, y1],
                # The image itself as text.
                "jpg": base64.b64encode(jpg.tobytes()).decode(),
            })
    # Everything the page needs, as one JSON block.
    data = {"video": info.path.name, "points": POINTS, "frames": frames}
    # Fill the template and write the page.
    page = out / "index.html"
    # The data replaces a placeholder inside the page's script.
    page.write_text(PAGE.replace("__DATA__", json.dumps(data)))
    # Return the path so the caller can print it.
    return page


# The marking page: plain HTML, CSS and JavaScript in one file, no internet needed.
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>batlab marking</title>
<style>
/* Colours used throughout the page. */
:root { --bg:#111418; --panel:#1b2027; --text:#e8ecf1; --muted:#9aa4b1; --accent:#7bd389; --warn:#f2c14e; }
/* Width and height include padding and borders, so layouts add up. */
* { box-sizing:border-box; }
/* Dark page with a system font. */
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 -apple-system,system-ui,sans-serif; }
/* Two columns: the frame on the left, instructions on the right. */
main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; padding:16px; height:100vh; }
/* The frame area; scrolls if the image is larger than the window. */
#stage { position:relative; display:flex; align-items:flex-start; justify-content:center; overflow:auto; }
/* Holds the image and the dots placed on top of it. */
#wrap { position:relative; }
/* The frame fits the window height; a crosshair pointer for precise clicks. */
#img { display:block; max-height:calc(100vh - 32px); max-width:100%; cursor:crosshair; user-select:none; }
/* A marked point: green circle centred on the click. */
.dot { position:absolute; width:14px; height:14px; margin:-7px 0 0 -7px; border:2px solid #fff; border-radius:50%;
       background:var(--accent); pointer-events:none; }
/* The point's number beside its dot, readable on any background. */
.dot span { position:absolute; left:16px; top:-4px; font-size:12px; white-space:nowrap; color:#fff;
            text-shadow:0 0 3px #000, 0 0 3px #000; }
/* The instructions panel. */
aside { background:var(--panel); border-radius:10px; padding:16px; overflow:auto; }
/* Panel heading and small grey text. */
h1 { font-size:17px; margin:0 0 4px; }
.muted { color:var(--muted); font-size:13px; }
/* The box saying which point to click next. */
#now { margin:14px 0; padding:12px; border-radius:8px; background:#252c35; font-weight:600; }
/* Checklist: green when marked, yellow when skipped, bold for the next one. */
ol { padding-left:20px; margin:8px 0; } li { margin:3px 0; } li.done { color:var(--accent); } li.skip { color:var(--warn); }
li.cur { font-weight:700; }
/* The square zoomed view. */
#loupe { width:100%; aspect-ratio:1; border-radius:8px; background:#000; display:block; margin:10px 0; }
/* Buttons; the download button stands out in green. */
button { background:#2d3540; color:var(--text); border:1px solid #3a4452; border-radius:6px; padding:7px 10px;
         font:inherit; cursor:pointer; margin:3px 2px 3px 0; }
button.primary { background:var(--accent); color:#0c1a10; border:none; font-weight:700; }
/* Keyboard shortcut labels. */
kbd { background:#2d3540; border-radius:4px; padding:1px 5px; font-size:12px; }
/* On narrow screens, stack the panel under the frame. */
@media (max-width: 800px) { main { grid-template-columns:1fr; height:auto; } }
</style></head><body>
<main>
  <div id="stage"><div id="wrap"><img id="img" alt="frame to mark"></div></div>
  <aside>
    <h1>Mark the batter</h1>
    <div class="muted" id="where"></div>
    <div id="now"></div>
    <canvas id="loupe" width="300" height="300"></canvas>
    <div class="muted">Zoomed 4x around the pointer. Click the exact spot.</div>
    <ol id="list"></ol>
    <div>
      <button id="undo">Undo <kbd>Z</kbd></button>
      <button id="skip">Can't see it <kbd>S</kbd></button>
    </div>
    <div>
      <button id="prev">&larr; Frame <kbd>P</kbd></button>
      <button id="next">Frame &rarr; <kbd>N</kbd></button>
    </div>
    <p class="muted" id="count"></p>
    <button class="primary" id="save">Download labels.json</button>
    <p class="muted">Progress is kept in this browser if you close the page. When done, put
    labels.json in the same folder as this page.</p>
  </aside>
</main>
<script>
// The frames and point list, filled in by Python when the page is built.
const DATA = __DATA__;
// Name under which progress is saved in this browser, one per video.
const KEY = "batlab-labels-" + DATA.video;
// labels[i][pointKey] = [x, y] in full-frame pixels, or null for "can't see it". One object per frame.
let labels = DATA.frames.map(() => ({}));
// Restore earlier progress if there is any; try/catch because browser storage can be unavailable.
try {
  // Read and parse the saved progress.
  const saved = JSON.parse(localStorage.getItem(KEY) || "null");
  // Use it only if it matches this page's number of frames.
  if (saved && saved.length === labels.length) labels = saved;
} catch (e) {}
// Index of the frame on screen.
let fi = 0;
// The image element and the element that holds the dots.
const img = document.getElementById("img"), wrap = document.getElementById("wrap");
// Drawing surface for the zoomed view.
const loupe = document.getElementById("loupe").getContext("2d");

// Save progress in this browser (ignored if storage is unavailable).
function store() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }

// The first point not yet marked or skipped on this frame, or undefined when all are done.
function nextPoint() { return DATA.points.find(([k]) => !(k in labels[fi])); }

// Redraw everything for the current frame.
function render() {
  // The current frame's data.
  const f = DATA.frames[fi];
  // Load the image only when the frame changes (reloading on every redraw would flicker).
  if (img.dataset.fi != fi) { img.src = "data:image/jpeg;base64," + f.jpg; img.dataset.fi = fi; }
  // Show which frame this is.
  document.getElementById("where").textContent =
    `Frame ${fi + 1} of ${DATA.frames.length} · delivery ${f.delivery} · ` +
    (f.frames_before_release ? `${f.frames_before_release} frames before release` : "release frame");
  // The point to click next.
  const np = nextPoint();
  // Show the instruction, or a note that this frame is finished.
  document.getElementById("now").textContent = np ? "Click: " + np[1] : "All points done for this frame. Press N for the next.";
  // Rebuild the checklist, colouring done, skipped and current points.
  document.getElementById("list").innerHTML = DATA.points.map(([k, t]) => {
    // This point's saved value (undefined = not yet marked).
    const v = labels[fi][k];
    // CSS class: "cur" for the next point, "done" if marked, "skip" if skipped.
    const cls = v === undefined ? (np && np[0] === k ? "cur" : "") : (v === null ? "skip" : "done");
    // One list item.
    return `<li class="${cls}">${t}${v === null ? " (skipped)" : ""}</li>`;
  }).join("");
  // Remove the old dots.
  wrap.querySelectorAll(".dot").forEach(d => d.remove());
  // Draw a numbered dot for each marked point.
  DATA.points.forEach(([k], i) => {
    // Saved position (skip missing or skipped points).
    const v = labels[fi][k]; if (!v) return;
    // Create the dot element.
    const d = document.createElement("div"); d.className = "dot";
    // Convert full-frame pixels back to on-screen position inside the shown image.
    d.style.left = ((v[0] - f.offset[0]) / img.naturalWidth * img.clientWidth) + "px";
    // Same for the vertical position.
    d.style.top = ((v[1] - f.offset[1]) / img.naturalHeight * img.clientHeight) + "px";
    // Label the dot with the point's number in the list.
    d.innerHTML = `<span>${i + 1}</span>`; wrap.appendChild(d);
  });
  // Count frames where every point is marked or skipped.
  const done = labels.filter(l => DATA.points.every(([k]) => k in l)).length;
  // Show the count.
  document.getElementById("count").textContent = `${done} of ${DATA.frames.length} frames complete`;
}

// Redraw once the image has loaded, because dot positions depend on its size.
img.addEventListener("load", render);
// A click marks the next point.
img.addEventListener("click", e => {
  // Nothing to do if this frame is finished.
  const np = nextPoint(); if (!np) return;
  // Current frame's data.
  const f = DATA.frames[fi];
  // Screen position -> pixel in the crop -> pixel in the full frame.
  const x = e.offsetX / img.clientWidth * img.naturalWidth + f.offset[0];
  // Same for y.
  const y = e.offsetY / img.clientHeight * img.naturalHeight + f.offset[1];
  // Save, rounded to 0.1 px, then store and redraw.
  labels[fi][np[0]] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10]; store(); render();
});
// Moving the pointer updates the 4x zoomed view.
img.addEventListener("mousemove", e => {
  // Pointer position in crop pixels.
  const sx = e.offsetX / img.clientWidth * img.naturalWidth, sy = e.offsetY / img.clientHeight * img.naturalHeight;
  // Keep pixels sharp when enlarged, and clear the view.
  loupe.imageSmoothingEnabled = false; loupe.fillStyle = "#000"; loupe.fillRect(0, 0, 300, 300);
  // Copy a 75 x 75 px area around the pointer into the 300 x 300 view (4x).
  loupe.drawImage(img, sx - 37.5, sy - 37.5, 75, 75, 0, 0, 300, 300);
  // Draw a small green cross at the centre: the exact spot a click would mark.
  loupe.strokeStyle = "#7bd389"; loupe.beginPath(); loupe.moveTo(150, 135); loupe.lineTo(150, 165);
  // Horizontal stroke of the cross.
  loupe.moveTo(135, 150); loupe.lineTo(165, 150); loupe.stroke();
});

// Remove the most recent mark on this frame.
function undo() {
  // Marked or skipped points, in list order.
  const ks = DATA.points.map(([k]) => k).filter(k => k in labels[fi]);
  // Delete the last one, then store and redraw.
  if (ks.length) { delete labels[fi][ks[ks.length - 1]]; store(); render(); }
}
// Record the next point as not visible.
function skip() { const np = nextPoint(); if (np) { labels[fi][np[0]] = null; store(); render(); } }
// Move d frames forward (or back), staying within the list.
function go(d) { fi = Math.min(DATA.frames.length - 1, Math.max(0, fi + d)); render(); }

// Connect the buttons.
document.getElementById("undo").onclick = undo; document.getElementById("skip").onclick = skip;
// Frame navigation buttons.
document.getElementById("prev").onclick = () => go(-1); document.getElementById("next").onclick = () => go(1);
// Keyboard shortcuts: Z undo, S skip, N next frame, P previous frame.
document.addEventListener("keydown", e => {
  // Lowercase so Shift does not matter.
  const k = e.key.toLowerCase();
  // Run the matching action.
  if (k === "z") undo(); else if (k === "s") skip(); else if (k === "n") go(1); else if (k === "p") go(-1);
});
// Download all marks as labels.json.
document.getElementById("save").onclick = () => {
  // The file's contents: video name, point order, and each frame's marks (images left out).
  const out = { video: DATA.video, points: DATA.points.map(([k]) => k),
    frames: DATA.frames.map((f, i) => ({ delivery: f.delivery, frame: f.frame,
      frames_before_release: f.frames_before_release, points: labels[i] })) };
  // A temporary link pointing at the file contents.
  const a = document.createElement("a");
  // Turn the JSON text into a downloadable file.
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }));
  // File name, then trigger the download.
  a.download = "labels.json"; a.click();
};
// Dots are positioned in screen pixels, so redraw when the window size changes.
window.addEventListener("resize", render);
// First draw.
render();
</script></body></html>
"""
