using app.core.data;
using ScottPlot.Avalonia;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Controls;
using System.IO;

namespace app.core.visualizer
{
    public class InformativenessVisualizerTxt : IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness, Window window, string outputPath)
        {
            double[] xs = Enumerable.Range(1, informativeness.informativenessList.Count)
                                    .Select(i => (double)i).ToArray();
            double[] ys = informativeness.informativenessList.ToArray();

            var plotCtrl = window.FindControl<AvaPlot>("PlotControl");

            if (plotCtrl == null)
                return;

            // save to file

            string dataFileName = $"informativeness_{informativeness.metricName}_{DateTime.Now.ToString("yyyy_MM_dd_hh_mm_ss")}.txt";
            string dataFilePath = Path.Combine(outputPath, dataFileName);

            int maxFeatureNameLength = informativeness.featureData.nameList.Max(name => name.Length);

            List<string> informativesnssResultToFileData = new List<string>();
            for (int i = 0; i < ys.Length; i++)
            {
                string featureName = (informativeness.featureData.nameList == null || informativeness.featureData.nameList.Count < i + 1) ? "" : informativeness.featureData.nameList[i + 1];
                informativesnssResultToFileData.Add(String.Format("{0,-" + (maxFeatureNameLength + 2) + ":g} {1,6:f2}", featureName, ys[i]));
            }
            File.WriteAllLines(dataFilePath, informativesnssResultToFileData);

            // log

            var savedLabel = window.FindControl<TextBlock>("InformativenessSavedLabel");
            if (savedLabel != null)
            {
                savedLabel.Text = $"Informativeness values saved in: {dataFilePath}";
                savedLabel.IsVisible = true;
            }
        }


    }
}
