using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using calculator.distance;
using calculator.informativeness;
using code.data;
using data;
using loader;
//using ScottPlot;

// Пример использования
class Program
{
    static void Main()
    {
        IDataLoader dataLoader = new DataLoadTxt();

        var data = dataLoader.LoadData("./resources/Th.txt");

        IInformativenessCalculator informativenessCalculator = new InformativenessCalculatorDefault();
        InformativenessCalculationResult informativeness = informativenessCalculator.Calculate(data, new DistanceCalculatorEuclidean());

        printTable(informativeness.informativenessList);
        //calculator.PlotResults(info);
    }

    // Вывод таблицы
    public static void printTable(List<double> informativeness)
    {
        Console.WriteLine("Номер\tИнформативность");
        for (int i = 0; i < informativeness.Count; i++)
            Console.WriteLine($"{i + 1}\t{informativeness[i]:F2}");
    }

    //// Построение графика
    //public void PlotResults(double[] informativeness, string outputPath = "informativeness.png")
    //{
    //    var plt = new Plot(800, 400);
    //    plt.Title("Информативность признаков");
    //    plt.XLabel("Номер признака");
    //    plt.YLabel("Коэффициент информативности");

    //    double[] positions = Enumerable.Range(1, informativeness.Length).Select(x => (double)x).ToArray();
    //    plt.AddBar(informativeness, positions);
    //    plt.SaveFig(outputPath);
    //}
}