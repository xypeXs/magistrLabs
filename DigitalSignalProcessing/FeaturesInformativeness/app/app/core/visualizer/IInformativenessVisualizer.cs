using app.core.data;

namespace app.core.visualizer
{
    public interface IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness, string outputPath, string name);
    }
}