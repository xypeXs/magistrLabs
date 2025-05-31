using app.core.data;
using ScottPlot;
using System.IO;
using System.Linq;
using System.Security.Principal;
using Avalonia.Controls;
using ScottPlot.Avalonia;
using static SkiaSharp.HarfBuzz.SKShaper;
using System;

namespace app.core.visualizer
{
    public class InformativenessVisualizerPlot : IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness, Window window, string outputPath)
        {
            double[] xs = Enumerable.Range(1, informativeness.informativenessList.Count)
                                    .Select(i => (double)i).ToArray();
            double[] ys = informativeness.informativenessList.ToArray();

            var plotCtrl = window.FindControl<AvaPlot>("PlotControl");

            if (plotCtrl == null)
                return;

            plotCtrl.Plot.Clear();
            plotCtrl.Plot.Title("Информативность признаков");
            plotCtrl.Plot.XLabel("Номер признака");
            plotCtrl.Plot.YLabel("Коэффициент информативности");

            plotCtrl.Plot.Add.Scatter(xs, ys);

            double xMax = xs.Max();
            double yMax = ys.Max();
            double xMin = xs.Min();
            double yMin = ys.Min();

            plotCtrl.Plot.Axes.SetLimits(xMin - 2, xMax + 2, yMin - 2, yMax + 2);

            plotCtrl.Refresh();

            // save to file

            string metricName = informativeness.metricName;
            string fileName = $"informativeness_{metricName}_{DateTime.Now.ToString("yyyy_MM_dd_hh_mm_ss")}.png";
            string fullPath = Path.Combine(outputPath, fileName);
            plotCtrl.Plot.SavePng(fullPath, 1920, 1080);

            // log

            var infoLabel = window.FindControl<TextBlock>("InfoLabel");
            if (infoLabel != null)
                infoLabel.Text = $"Image saved at: {fullPath}";
        }
    }
}
