"""Record short GIF demos for each FlashStudio page."""

import asyncio
import os
import glob
import numpy as np
import imageio.v3 as iio
from PIL import Image
from playwright.async_api import async_playwright

APP_URL = "http://127.0.0.1:8510"
OUT_DIR = os.path.join(os.path.dirname(__file__), "demos")
FRAME_DIR = os.path.join(OUT_DIR, "_frames")
os.makedirs(FRAME_DIR, exist_ok=True)

VIEWPORT = {"width": 1440, "height": 900}
GIF_DURATION_MS = 1200

SANITIZE_JS = """
() => {
    function sanitize(t) {
        t = t.replace(/\\/home\\/[^\\/]+\\//g, '');
        t = t.replace(/\\/Users\\/[^\\/]+\\//g, '');
        t = t.replace(/\\/root\\//g, '');
        t = t.replace(/[A-Z]:\\\\Users\\\\[^\\\\]+\\\\/g, '');
        t = t.replace(/\\.?flashstudio\\/projects\\/[^\\/]+\\//g, 'workspace/');
        t = t.replace(/\\.?flashstudio\\//g, 'workspace/');
        t = t.replace(/\\.?\\/?(?:Project|Projects|Dev|Code|Work|repos|src)\\/[^\\/]+(?:\\/[^\\/]+)*\\//gi,
            (match) => {
                const parts = match.replace(/^\\.\\//, '').split('/').filter(Boolean);
                if (parts.length > 2) return parts.slice(-1)[0] + '/';
                return parts.join('/') + '/';
            });
        t = t.replace(/^\\.\\//, '');
        t = t.replace(/\\/\\//g, '/');
        return t;
    }
    const walk = (node) => {
        if (node.nodeType === 3 && node.textContent.trim()) {
            node.textContent = sanitize(node.textContent);
        }
        for (const child of node.childNodes) walk(child);
    };
    walk(document.body);
    document.querySelectorAll('input, textarea').forEach(el => {
        if (el.value) el.value = sanitize(el.value);
    });
    document.querySelectorAll('[data-testid] input').forEach(el => {
        if (el.value) {
            const nativeSet = Object.getOwnPropertyDescriptor(
                HTMLInputElement.prototype, 'value').set;
            nativeSet.call(el, sanitize(el.value));
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }
    });
}
"""


async def screenshot(page, name, wait_ms=800):
    await page.wait_for_timeout(wait_ms)
    await page.evaluate(SANITIZE_JS)
    await page.wait_for_timeout(100)
    path = os.path.join(FRAME_DIR, f"{name}.png")
    await page.screenshot(path=path, full_page=False)
    print(f"  captured: {name}")
    return path


async def nav_to_page(page, index, label):
    btn = page.locator(f'button:has-text("{label}")').first
    await btn.click()
    await page.wait_for_timeout(2000)


async def click_tab(page, tab_name):
    tabs = page.locator('[role="tab"]')
    count = await tabs.count()
    for i in range(count):
        txt = (await tabs.nth(i).inner_text()).strip()
        if tab_name in txt:
            await tabs.nth(i).click()
            await page.wait_for_timeout(1200)
            return True
    return False


def frames_to_gif(prefix, output_name, duration_ms=GIF_DURATION_MS):
    pattern = os.path.join(FRAME_DIR, f"{prefix}_*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no frames for {prefix}")
        return

    raw_frames = []
    for f in files:
        raw_frames.append(np.array(Image.open(f).convert("RGB")))

    # Deduplicate identical frames
    frames = [raw_frames[0]]
    for i in range(1, len(raw_frames)):
        if not np.array_equal(raw_frames[i], raw_frames[i - 1]):
            frames.append(raw_frames[i])

    if len(frames) < 2:
        frames = raw_frames[:2] if len(raw_frames) >= 2 else raw_frames

    out_path = os.path.join(OUT_DIR, f"{output_name}.gif")
    durations = [duration_ms] * len(frames)
    durations[-1] = duration_ms * 2

    iio.imwrite(out_path, frames, duration=durations, loop=0, plugin="pillow")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  -> {output_name}.gif ({len(frames)} unique frames, {size_mb:.1f}MB)")


async def record_dashboard(page):
    print("\n[1/6] Dashboard")
    await nav_to_page(page, 0, "Dashboard")
    await screenshot(page, "dashboard_01_overview", 1500)

    # Click Quick Action buttons to highlight interactivity
    upload_btn = page.locator('button:has-text("Upload Data")').first
    if await upload_btn.count() > 0:
        await upload_btn.hover()
        await screenshot(page, "dashboard_02_hover_upload", 400)

    train_btn = page.locator('button:has-text("Start Training")').first
    if await train_btn.count() > 0:
        await train_btn.hover()
        await screenshot(page, "dashboard_03_hover_train", 400)

    infer_btn = page.locator('button:has-text("Run Inference")').first
    if await infer_btn.count() > 0:
        await infer_btn.hover()
        await screenshot(page, "dashboard_04_hover_infer", 400)

    frames_to_gif("dashboard", "01_dashboard")


async def record_data(page):
    print("\n[2/6] Data")
    await nav_to_page(page, 1, "Data")
    await screenshot(page, "data_01_upload_tab", 1500)
    await click_tab(page, "Download")
    await screenshot(page, "data_02_download_tab", 800)
    await click_tab(page, "Preview")
    await screenshot(page, "data_03_preview_tab", 800)
    await click_tab(page, "Verify")
    await screenshot(page, "data_04_verify_tab", 800)
    frames_to_gif("data", "02_data")


async def record_model(page):
    print("\n[3/6] Model")
    await nav_to_page(page, 2, "Model")
    await screenshot(page, "model_01_architecture", 1500)
    await click_tab(page, "Hyperparams")
    await screenshot(page, "model_02_hyperparams", 800)
    await click_tab(page, "Augment")
    await screenshot(page, "model_03_augment", 800)
    await click_tab(page, "Advanced")
    await screenshot(page, "model_04_advanced", 800)
    frames_to_gif("model", "03_model")


async def record_training(page):
    print("\n[4/6] Training")
    await nav_to_page(page, 3, "Training")
    await screenshot(page, "training_01_launch", 1500)

    await click_tab(page, "Monitor")
    await screenshot(page, "training_02_monitor", 1000)

    # Go back to Launch to show experiment section
    await click_tab(page, "Launch")
    await screenshot(page, "training_03_experiment", 1000)

    frames_to_gif("training", "04_training")


async def record_export(page):
    print("\n[5/6] Export")
    await nav_to_page(page, 4, "Export")
    await screenshot(page, "export_01_overview", 1500)

    # Hover over Export button
    export_btn = page.locator('button:has-text("Export ONNX")').first
    if await export_btn.count() > 0:
        await export_btn.hover()
        await screenshot(page, "export_02_hover_export", 600)

    frames_to_gif("export", "05_export")


async def record_inference(page):
    print("\n[6/6] Inference")
    await nav_to_page(page, 5, "Inference")
    await screenshot(page, "inference_01_model_tab", 1500)

    await click_tab(page, "Data")
    await screenshot(page, "inference_02_data_tab", 800)

    await click_tab(page, "Solution")
    await screenshot(page, "inference_03_solution", 1200)

    # Click Rectangle radio to show mode switching
    rect_radio = page.locator('label:has-text("Rectangle")').first
    if await rect_radio.count() > 0:
        await rect_radio.click()
        await page.wait_for_timeout(800)
        await screenshot(page, "inference_04_rect_mode", 600)

    # Click Line radio
    line_radio = page.locator('label:has-text("Line")').first
    if await line_radio.count() > 0:
        await line_radio.click()
        await page.wait_for_timeout(800)
        await screenshot(page, "inference_05_line_mode", 600)

    await click_tab(page, "Run")
    await screenshot(page, "inference_06_run_tab", 800)

    frames_to_gif("inference", "06_inference")


async def main():
    print("=" * 50)
    print("FlashStudio UI Demo Recorder")
    print("=" * 50)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        print("\nConnecting...")
        await page.goto(APP_URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)
        print("Connected!")

        await record_dashboard(page)
        await record_data(page)
        await record_model(page)
        await record_training(page)
        await record_export(page)
        await record_inference(page)

        await browser.close()

    print("\n" + "=" * 50)
    print("All demos saved to: demos/")
    gifs = sorted(glob.glob(os.path.join(OUT_DIR, "*.gif")))
    for g in gifs:
        size = os.path.getsize(g) / 1e6
        print(f"  {os.path.basename(g):30s} {size:.1f}MB")
    print(f"\nIndividual frames: demos/_frames/")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
