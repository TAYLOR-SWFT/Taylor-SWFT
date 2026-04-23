#let stat_dir = "stats/"
#let color_best = rgb("#176b17")
#let color_worst = rgb("#c5ffc4")
#let rooms = (
  "CR1",
  "CR2",
  "CR3",
  "CR4",
)
#let baselines = (
  "taylor_swft",
  "ism_rt",
  "rt",
  "ism",
  "noise",
)
#let metrics = (
  "clarity_50ms",
  "definition_50ms",
  "direct_to_reverberant_ratio",
  "reverb_time_30_dB",
  "energy_decay_curve",
  "energy_decay_relief",
  "mel_energy_decay_relief",
  "dtw",
  "computation_time",
)
#let pretty_rooms = (CR1: "Coupled Rooms", CR2: "Seminar Room", CR3: "Chamber Music Hall", CR4: "Auditorium")
#let pretty_baselines = (ism: "ISM", rt: "RT", ism_rt: "ISM-RT", noise: "Noise", taylor_swft: "T-SWFT")
#let pretty_metrics = (
  clarity_50ms: [$arrow.b Delta C_50$ (dB)],
  definition_50ms: [$arrow.b Delta D_50$ (%)],
  direct_to_reverberant_ratio: [$arrow.b Delta D R R$ (dB)],
  reverb_time_30_dB: [$arrow.b Delta R T_30$ (s)],
  energy_decay_curve: [$arrow.b Delta E D C$ (dB)],
  energy_decay_relief: [$arrow.b Delta E D R$ (dB)],
  mel_energy_decay_relief: [$arrow.b Delta "Mel" E D R$ (dB)],
  computation_time: [$arrow.b$ Computation Time (s)],
  dtw: [$arrow.b$ DTW],
)


#let fmt(val, precision: 2) = {
  let n = calc.round(float(val), digits: precision)
  str(n)
}

#let format-cell(mean, std, is_best: "False", is_worst: "False") = {
  let m = float(mean)
  let precision
  if m >= 100 {
    precision = 0
  } else if m >= 10 {
    precision = 1
  } else {
    precision = 2
  }
  if (is_best == "True") {
    table.cell(fill: color_best)[#text(
      rgb("#ffffff"),
      $#fmt(mean, precision: precision) plus.minus #fmt(std, precision: precision)$,
    )]
  } else if (is_worst == "True") {
    table.cell(fill: color_worst)[$#fmt(mean, precision: precision) plus.minus #fmt(std, precision: precision)$]
  } else {
    [$#fmt(mean, precision: precision) plus.minus #fmt(std, precision: precision)$]
  }
}

#figure(
  caption: [
    Comparison of the methods. Each cell corresponds to the mean $plus.minus$ standard deviation over 209 RIRs. The lower is better for all metrics.
  ],
  // scope: "parent",
  placement: auto, // top, bottom or none for "right here"
  text(size: 10pt, table(
    row-gutter: 0pt,
    inset: 2.5pt,
    columns: (auto, ..(1fr,) * rooms.len()),
    stroke: 0.3pt + gray,

    // Table Header
    table.header([], ..rooms.map(r => [*#pretty_rooms.at(r)*])),

    // Loop through each Metric (Main Rows)
    ..for metric in metrics {
      (
        // The "Main Row" spanning all columns
        table.cell(colspan: rooms.len() + 1, fill: luma(240))[*#pretty_metrics.at(metric)*],
        // Loop through Baselines (Sub-rows)
        ..for base in baselines {
          (
            [#pretty_baselines.at(base)], // Baseline Label
            // Loop through Rooms (Columns)
            ..rooms.map(room => {
              // Dynamically load the specific file for this Room/Baseline pair
              let file-path = stat_dir + "stats_" + base + "_" + room + ".csv"
              let data = csv(file-path)

              // Find the row for this metric
              let target = data.find(r => r.at(0) == metric)

              if target != none {
                // target.at(1) is mean, target.at(2) is std
                format-cell(target.at(1), target.at(2), is_best: target.at(3), is_worst: target.at(5))
              } else [---]
            }),
          )
        },
      )
    },
  )),
)
