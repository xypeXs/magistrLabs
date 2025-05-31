using app.core.data;
using Avalonia.Controls;

namespace app.core.visualizer
{
    public interface IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness, Window window, string outputPath);
    }
}