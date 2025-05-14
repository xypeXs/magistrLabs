// MainWindow.axaml.cs
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media.Imaging;
using Avalonia.Interactivity;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System;
using core.loader;
using core.data;
using core.calculator.distance;
using core.calculator.informativeness;
using core.visualizer;

namespace app.Views;

public partial class MainWindow : Window
{

    private String inputPath;
    private String outputPath;
    private List<IDistanceCalculator> informativenessCalculators = new List<IDistanceCalculator>
    {
        new DistanceCalculatorEuclidean(),
        new DistanceCalculatorChebyshev(),
        new DistanceCalculatorMahalanobis(),
        new DistanceCalculatorCanberra()
    };


    public MainWindow()
    {
        InitializeComponent();
    }

    private async void LoadFile_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog();
        dialog.Filters.Add(new FileDialogFilter() { Name = "All Files", Extensions = { "*" } });
        dialog.AllowMultiple = false;

        var result = await dialog.ShowAsync(this);

        if (result == null || result.Length <= 0)
        {
            return;
        }

        this.inputPath = result[0];
    }

    private async void ChooseOutput_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog();

        var result = await dialog.ShowAsync(this);

        if (result == null || result.Length <= 0)
        {
            return;
        }

        this.outputPath = result;
    }

    private async void Proceed_Click(object sender, RoutedEventArgs e)
    {
        var extension = Path.GetExtension(inputPath).ToLower();
        IDataLoader dataLoader = GetDataLoader(inputPath);
        FeaturesData featuresData = dataLoader.LoadData(inputPath);

        string metricSysName = ((ComboBoxItem) Metric_combo_box.SelectedItem).Content.ToString();
        IDistanceCalculator distanceCalculator = informativenessCalculators.Where(calculator => calculator.GetType() == metricSysName).First();
        distanceCalculator.initialize(featuresData);

        IInformativenessCalculator informativenessCalculator = new InformativenessCalculatorDefault();
        InformativenessCalculationResult informativenessEuclidian = informativenessCalculator.Calculate(featuresData, distanceCalculator);

        IInformativenessVisualizer informativenessVisualizer = new InformativenessVisualizerPlot();
        informativenessVisualizer.visualize(informativenessEuclidian, outputPath, distanceCalculator.GetType());
    }

    protected IDataLoader GetDataLoader(string filePath)
    {
        // Хардкод
        return new DataLoadTxt();
    }
}
