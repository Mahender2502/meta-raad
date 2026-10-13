import { useState, useEffect } from "react";
import { NavLink, Outlet, useLocation, Link } from "react-router-dom";
import {
  Activity,
  ArrowUpRight,
  Bell,
  BookOpen,
  Boxes,
  ChartNoAxesCombined,
  ChevronDown,
  Database,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  Menu,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import { Button, Notice, Status } from "../components/common";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";
const navigation = [
  { path: "/", label: "Overview", icon: LayoutDashboard },
  { path: "/datasets", label: "Datasets", icon: Database },
  { path: "/vectors", label: "Vector Database", icon: Boxes },
  { path: "/detection", label: "Anomaly Detection", icon: Activity },
  { path: "/retrieval", label: "Retrieval Explorer", icon: Search },
  { path: "/models", label: "Model Selection", icon: GitBranch },
  { path: "/evaluation", label: "Evaluation", icon: ChartNoAxesCombined },
  { path: "/settings", label: "Settings", icon: Settings2 },
];
export function WorkspaceLayout() {
  const w = useWorkspace(),
    location = useLocation(),
    [mobile, setMobile] = useState(false),
    [notifications, setNotifications] = useState(false),
    [profile, setProfile] = useState(false);
  useEffect(() => {
    if (!mobile) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobile(false);
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [mobile]);
  const page =
    navigation.find((n) => n.path === location.pathname)?.label ?? "Workspace";
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      {mobile && (
        <button
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobile(false)}
        />
      )}
      <aside
        id="workspace-navigation"
        className={`sidebar ${mobile ? "open" : ""}`}
      >
        <Link to="/" className="brand" onClick={() => setMobile(false)}>
          <div className="brand-symbol">
            <Boxes size={25} />
          </div>
          <div>
            Meta<span>RAAD</span>
            <small>RESEARCH WORKSPACE</small>
          </div>
        </Link>
        <button
          className="mobile-close"
          aria-label="Close navigation"
          onClick={() => setMobile(false)}
        >
          <X size={18} />
        </button>
        <div className="workspace-chip">
          <span className="workspace-icon">
            <FlaskConical size={16} />
          </span>
          <div>
            Text anomaly research<small>Personal workspace</small>
          </div>
          <ChevronDown size={13} />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {navigation.map((n, i) => (
            <NavLink
              key={n.path}
              to={n.path}
              end={n.path === "/"}
              onClick={() => setMobile(false)}
              className={({ isActive }) =>
                `${isActive ? "active" : ""} ${i === 3 ? "primary-nav" : ""}`
              }
            >
              <n.icon size={18} />
              {n.label}
              {i === 3 && <span className="nav-pill">RA-ZAD</span>}
              {i === 5 && <span className="nav-pill">FG-MOS</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="research-note">
            <ShieldCheck size={19} />
            <strong>Evidence before conclusions</strong>
            <p>
              Training-only retrieval.
              <br />
              Traceable model decisions.
            </p>
            <a
              href="https://github.com/USC-FORTIS/AD-LLM"
              target="_blank"
              rel="noreferrer"
            >
              Base paper & code <ArrowUpRight size={13} />
            </a>
          </div>
          <div className="sidebar-dataset">
            <Database size={15} />
            <select
              aria-label="Sidebar dataset"
              value={w.dataset?.id ?? ""}
              onChange={(e) => w.selectDataset(e.target.value)}
            >
              {!w.datasets.length && (
                <option value="">No dataset loaded</option>
              )}
              {w.datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>
          <div className="sidebar-status">
            <i className={w.connected ? "online" : ""} />
            {w.preferences.mode === "demo"
              ? "Illustrative preview"
              : w.connected
                ? "Backend connected"
                : "Backend offline"}
            <span>v{w.catalog?.configuration.version ?? "0.2.0"}</span>
          </div>
        </div>
      </aside>
      <div className="app-main">
        <header className="topbar">
          <div className="breadcrumb">
            <Button
              variant="ghost"
              size="icon"
              className="menu-toggle"
              aria-label="Open navigation"
              aria-expanded={mobile}
              aria-controls="workspace-navigation"
              onClick={() => setMobile(true)}
            >
              <Menu />
            </Button>
            <span>Workspace</span>
            <span className="slash">/</span>
            <strong>{page}</strong>
          </div>
          <div className="topbar-actions">
            <div className="dataset-switch">
              <Database size={14} />
              <select
                aria-label="Selected dataset"
                value={w.dataset?.id ?? ""}
                onChange={(e) => w.selectDataset(e.target.value)}
              >
                {!w.datasets.length && <option value="">No dataset</option>}
                {w.datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
            <span className="topbar-divider" />
            <button
              className="icon-button"
              aria-label="Notifications"
              onClick={() => setNotifications(true)}
            >
              <Bell size={18} />
              {w.error && <i />}
            </button>
            <button
              className="avatar"
              aria-label="Workspace profile"
              onClick={() => setProfile(true)}
            >
              MR
            </button>
          </div>
        </header>
        <main id="main-content" className="content">
          {w.preferences.mode === "demo" && (
            <Notice tone="amber">
              <strong>Preview mode</strong> — All predictions, rankings,
              embeddings, and charts are illustrative. Uploaded files remain in
              this browser session.
            </Notice>
          )}
          {w.error && (
            <Notice tone="red">
              Backend unavailable: {w.error}.{" "}
              <Link to="/settings">Check connection settings</Link> or
              explicitly enable preview mode.
            </Notice>
          )}
          <div
            key={`${w.preferences.mode}:${w.preferences.backendUrl}:${w.dataset?.id}`}
          >
            <Outlet key={w.dataset?.id} />
          </div>
        </main>
        <footer className="workspace-footer">
          <span>
            <ShieldCheck size={13} /> Research integrity by design
          </span>
          <span>Meta-RAAD · Retrieval-augmented anomaly detection</span>
          <Link to="/settings">
            Connection settings <ArrowUpRight size={12} />
          </Link>
        </footer>
      </div>
      <Dialog open={notifications} onOpenChange={setNotifications}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Workspace activity</DialogTitle>
            <DialogDescription>
              Current session and backend connection.
            </DialogDescription>
          </DialogHeader>
          <div className="activity-line">
            <span className="activity-icon">
              <Database size={18} />
            </span>
            <div>
              <strong>
                {w.connected ? "Workspace ready" : "Connection needs attention"}
              </strong>
              <p>
                {w.error ||
                  `${w.datasets.length} datasets available · ${w.experiments.length} saved experiments`}
              </p>
            </div>
          </div>
          {w.latest && (
            <div className="activity-line">
              <Activity size={18} />
              <div>
                <strong>Latest detection complete</strong>
                <p>{new Date(w.latest.timestamp).toLocaleString()}</p>
              </div>
            </div>
          )}
          <Status tone={w.preferences.mode === "demo" ? "amber" : "green"}>
            {w.preferences.mode === "demo"
              ? "Illustrative data"
              : "Live backend data"}
          </Status>
        </DialogContent>
      </Dialog>
      <Dialog open={profile} onOpenChange={setProfile}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Meta-RAAD research workspace</DialogTitle>
            <DialogDescription>
              Local project workspace. No account or credentials are stored in
              the frontend.
            </DialogDescription>
          </DialogHeader>
          <p className="muted">
            Two research modules, one transparent workflow.
          </p>
          <div className="activity-line">
            <Sparkles />
            <div>
              <strong>RA-ZAD</strong>
              <p>Retrieval-augmented anomaly detection</p>
            </div>
          </div>
          <div className="activity-line">
            <GitBranch />
            <div>
              <strong>FG-MOS</strong>
              <p>Feature-grounded model selection</p>
            </div>
          </div>
          <Button asChild variant="outline">
            <Link to="/settings" onClick={() => setProfile(false)}>
              <BookOpen />
              Workspace settings
            </Link>
          </Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
