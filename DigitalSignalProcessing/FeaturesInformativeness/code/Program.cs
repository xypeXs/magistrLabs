using calculator.distance;
using calculator.informativeness;
using code.data;
using code.visualizer;
using loader;
using visualizer;

// Пример использования
class Program
{
    static void Main()
    {
        IDataLoader dataLoader = new DataLoadTxt();

        var data = dataLoader.LoadData("./resources/Th.txt");

        IInformativenessCalculator informativenessCalculator = new InformativenessCalculatorDefault();
        InformativenessCalculationResult informativenessEuclidian = informativenessCalculator.Calculate(data, new DistanceCalculatorEuclidean());
        InformativenessCalculationResult informativenessChebyshev = informativenessCalculator.Calculate(data, new DistanceCalculatorChebyshev());

        Console.WriteLine("Расстояние Евклида");
        printTable(informativenessEuclidian.informativenessList);
        
        Console.WriteLine("Расстояние Чебышёва");
        printTable(informativenessChebyshev.informativenessList);

        IInformativenessVisualizer informativenessVisualizer = new InformativenessVisualizerPlot();
        informativenessVisualizer.visualize(informativenessEuclidian);
    }

    // Вывод таблицы
    public static void printTable(List<double> informativeness)
    {
        Console.WriteLine("Номер\tИнформативность");
        for (int i = 0; i < informativeness.Count; i++)
            Console.WriteLine($"{i + 1}\t{informativeness[i]:F2}");
    }
}