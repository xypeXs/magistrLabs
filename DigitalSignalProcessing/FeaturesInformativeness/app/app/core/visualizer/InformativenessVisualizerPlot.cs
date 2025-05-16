using informativeness.app.core.data;
using ScottPlot;
using System.IO;
using System.Linq;

namespace informativeness.app.core.visualizer
{
	public class InformativenessVisualizerPlot : IInformativenessVisualizer
	{
		public void visualize(InformativenessCalculationResult informativeness, string outputPath, string name)
		{
			var plt = new Plot();
			plt.Title("Информативность признаков");
			plt.XLabel("Номер признака");
			plt.YLabel("Коэффициент информативности");

			double[] xs = Enumerable.Range(1, informativeness.informativenessList.Count).Select(i => (double)i).ToArray();
			double[] ys = informativeness.informativenessList.ToArray();

			var scatter = plt.Add.Scatter(xs, ys);
			scatter.MarkerSize = 5;
			scatter.MarkerShape = MarkerShape.FilledCircle;

			double xMax = xs.Max();
			double yMax = ys.Max();
			double xMin = xs.Min();
			double yMin = ys.Min();

			plt.Axes.SetLimits(xMin - 2, xMax + 2, yMin - 2, yMax + 2);

			string fileName = $"informativeness_{name}.bmp";
			string fullPath = Path.Combine(outputPath, fileName);
			plt.SaveBmp(fullPath, 1920, 1080);
		}
	}
}
