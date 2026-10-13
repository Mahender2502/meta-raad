import { lazy, Suspense } from "react";
import { Loading } from "./components/common";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { WorkspaceProvider } from "./hooks/use-workspace";
import { WorkspaceLayout } from "./layouts/workspace-layout";
const Overview = lazy(() =>
  import("./pages/overview").then((m) => ({ default: m.Overview })),
);
const Datasets = lazy(() =>
  import("./pages/datasets").then((m) => ({ default: m.Datasets })),
);
const Vectors = lazy(() =>
  import("./pages/vectors").then((m) => ({ default: m.Vectors })),
);
const DetectionPage = lazy(() =>
  import("./pages/detection").then((m) => ({ default: m.DetectionPage })),
);
const Retrieval = lazy(() =>
  import("./pages/retrieval").then((m) => ({ default: m.Retrieval })),
);
const Models = lazy(() =>
  import("./pages/models").then((m) => ({ default: m.Models })),
);
const Evaluation = lazy(() =>
  import("./pages/evaluation").then((m) => ({ default: m.Evaluation })),
);
const Settings = lazy(() =>
  import("./pages/settings").then((m) => ({ default: m.Settings })),
);
export default function App() {
  return (
    <BrowserRouter>
      <WorkspaceProvider>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route element={<WorkspaceLayout />}>
              <Route index element={<Overview />} />
              <Route path="datasets" element={<Datasets />} />
              <Route path="vectors" element={<Vectors />} />
              <Route path="detection" element={<DetectionPage />} />
              <Route path="retrieval" element={<Retrieval />} />
              <Route path="models" element={<Models />} />
              <Route path="evaluation" element={<Evaluation />} />
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<Overview />} />
            </Route>
          </Routes>
        </Suspense>
      </WorkspaceProvider>
    </BrowserRouter>
  );
}
