import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  AreaChart,
  Area,
  LineChart,
  Line,
} from "recharts";
const colors = { blue: "#5965da", teal: "#36a68c", gray: "#99a6b9" };
export function ComparisonChart({
  data,
}: {
  data: { name: string; auroc: number | null; auprc: number | null }[];
}) {
  return (
    <div
      className="chart"
      role="img"
      aria-label="AUROC and average precision by experiment variant"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          barGap={5}
          margin={{ top: 10, right: 10, left: -22, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            vertical={false}
            stroke="#e8edf4"
          />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11, fill: "#778399" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "#778399" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              borderRadius: 8,
              borderColor: "#e2e8f0",
              fontSize: 12,
            }}
          />
          <Legend
            iconType="circle"
            iconSize={7}
            wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
          />
          <Bar
            dataKey="auroc"
            name="AUROC"
            fill={colors.blue}
            radius={[4, 4, 0, 0]}
            maxBarSize={44}
          />
          <Bar
            dataKey="auprc"
            name="AUPRC (AP)"
            fill={colors.teal}
            radius={[4, 4, 0, 0]}
            maxBarSize={44}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
export function CurveChart({
  data,
  pr = false,
}: {
  data: { fpr: number; recall: number; precision: number }[];
  pr?: boolean;
}) {
  return (
    <div
      className="chart"
      role="img"
      aria-label={pr ? "Precision recall curve" : "ROC curve"}
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ left: -15, right: 10, bottom: 12, top: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            type="number"
            domain={[0, 1]}
            dataKey={pr ? "recall" : "fpr"}
            tick={{ fontSize: 11 }}
            label={{
              value: pr ? "Recall" : "False positive rate",
              position: "insideBottom",
              offset: -10,
              fontSize: 11,
            }}
          />
          <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Area
            type={pr ? "stepAfter" : "linear"}
            dataKey={pr ? "precision" : "recall"}
            name={pr ? "Precision" : "True positive rate"}
            stroke={colors.blue}
            fill="#eef0ff"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
export function SimilarityChart({
  data,
}: {
  data: { name: string; similarity: number }[];
}) {
  return (
    <div className="chart small-chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 0, right: 28, top: 0, bottom: 0 }}
        >
          <CartesianGrid horizontal={false} strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11 }}
            width={40}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip />
          <Bar
            dataKey="similarity"
            fill={colors.blue}
            radius={[0, 4, 4, 0]}
            barSize={20}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
export function GeometryChart({
  data,
}: {
  data: { name: string; value: number }[];
}) {
  return (
    <div className="chart small-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ left: -20, right: 20, top: 20, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 10 }} axisLine={false} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Line
            type="linear"
            dataKey="value"
            stroke={colors.blue}
            strokeWidth={2}
            dot={{ r: 5, fill: "white", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
