const REPO = "https://github.com/PjDailey11/neural-style-transfer";

export default function Page() {
  return (
    <main>
      <span className="badge">Vercel · project hub</span>
      <h1>Neural Style Transfer Toolkit</h1>
      <p className="lead">
        This site is the lightweight entry point for the PyTorch project. Heavy inference (VGG-19, AdaIN,
        optical flow, Streamlit UI) runs locally or in Docker — Vercel hosts documentation and links here,
        not the GPU stack.
      </p>

      <div className="card">
        <h2>What ships in the repo</h2>
        <p>
          Gatys optimization NST, feed-forward AdaIN, ReCoNet-style video nets, CLI scripts, Streamlit,
          Docker, and GitHub Actions CI.
        </p>
      </div>

      <div className="card">
        <h2>Run the full toolkit</h2>
        <p>
          Clone the repo, install Python deps (<code>pip install -e .</code>), use{" "}
          <code>transform_image.py</code> / <code>transform_video.py</code>, or{" "}
          <code>docker build</code> + <code>streamlit run streamlit_app.py</code>.
        </p>
      </div>

      <div className="actions">
        <a className="button primary" href={REPO} target="_blank" rel="noreferrer">
          View on GitHub
        </a>
        <a className="button" href={`${REPO}/blob/main/README.md`} target="_blank" rel="noreferrer">
          README
        </a>
        <a className="button" href={`${REPO}/actions`} target="_blank" rel="noreferrer">
          CI status
        </a>
      </div>

      <footer>
        MIT licensed · PyTorch + Next.js on Vercel for static routing only · Issues welcome on GitHub.
      </footer>
    </main>
  );
}
