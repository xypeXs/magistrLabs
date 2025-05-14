using core.data;

namespace core.visualizer
{
    public interface IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness, string outputPath, string name);
    }
}