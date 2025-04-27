using code.data;
using ScottPlot;

namespace visualizer
{
    public class InformativenessVisualizerPlot : IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness)
        {
            Plot plt = new Plot();
            plt.Title("Информативность признаков");
            plt.XLabel("Номер признака");
            plt.YLabel("Коэффициент информативности");

            double[] positions = Enumerable.Range(1, informativeness.informativenessList.Count).Select(x => (double)x).ToArray();
            plt.Add.Scatter(positions, informativeness.informativenessList.ToArray());
            plt.SaveBmp("informativeness.png", 1920, 1080);
        }
    }
}
