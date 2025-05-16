using calculator.distance;
using calculator.informativeness;
using code.data;
using loader;
using visualizer;

// Пример использования
class Program
{
    static void Main()
    {
        IDataLoader dataLoader = new DataLoadTxt();

        var data = dataLoader.LoadData("./resources/Th3.txt");
        
        IInformativenessCalculator informativenessCalculator = new InformativenessCalculatorDefault();
        InformativenessCalculationResult informativenessEuclidian = informativenessCalculator.Calculate(data, new DistanceCalculatorEuclidean());
        InformativenessCalculationResult informativenessChebyshev = informativenessCalculator.Calculate(data, new DistanceCalculatorChebyshev());
		InformativenessCalculationResult informativenessCanberra = informativenessCalculator.Calculate(data, new DistanceCalculatorCanberra());
		InformativenessCalculationResult informativenessMahalanobis = informativenessCalculator.Calculate(data, new DistanceCalculatorMahalanobis(data));

		Console.WriteLine("Расстояние Евклида");
        printTable(informativenessEuclidian.informativenessList);
        
        Console.WriteLine("Расстояние Чебышёва");
        printTable(informativenessChebyshev.informativenessList);

		Console.WriteLine("Расстояние Канберры");
		printTable(informativenessCanberra.informativenessList);

		Console.WriteLine("Расстояние Махаланобиса");
		printTable(informativenessMahalanobis.informativenessList);

		IInformativenessVisualizer informativenessVisualizer = new InformativenessVisualizerPlot();
        informativenessVisualizer.visualize(informativenessEuclidian, "euclidian");
		informativenessVisualizer.visualize(informativenessChebyshev, "chebushev");
		informativenessVisualizer.visualize(informativenessCanberra, "canberra");
		informativenessVisualizer.visualize(informativenessMahalanobis, "mahalanobis");
	}

    // Вывод таблицы
    public static void printTable(List<double> informativeness)
    {
        Console.WriteLine("Номер\tИнформативность");
        for (int i = 0; i < informativeness.Count; i++)
            Console.WriteLine($"{i + 1}\t{informativeness[i]:F2}");
    }
}