using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using informativeness.app.core.calculator.distance;
using informativeness.app.core.calculator.informativeness;
using informativeness.app.core.data;
using informativeness.app.core.loader;
using informativeness.app.core.visualizer;
using ScottPlot;
using ScottPlot.Avalonia;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Emit;

namespace informativeness.app.Views
{
	public partial class MainWindow : Window
	{
		private string inputPath;
		private string outputPath;
		private readonly List<IDistanceCalculator> calculators = new()
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
			var dlg = new OpenFileDialog
			{
				Filters = { new FileDialogFilter { Name = "All Files", Extensions = { "*" } } },
				AllowMultiple = false
			};
			var res = await dlg.ShowAsync(this);
			if (res?.Length > 0)
			{
				inputPath = res[0];
				InputPathLabel.Text = inputPath;
			}
		}

		private async void ChooseOutput_Click(object sender, RoutedEventArgs e)
		{
			var dlg = new OpenFolderDialog();
			var res = await dlg.ShowAsync(this);
			if (!string.IsNullOrEmpty(res))
			{
				outputPath = res;
				OutputPathLabel.Text = outputPath;
			}
		}

		private async void Proceed_Click(object sender, RoutedEventArgs e)
		{
			if (string.IsNullOrEmpty(inputPath) || string.IsNullOrEmpty(outputPath))
			{
				var alert = new Window
				{
					Title = "Внимание",
					Width = 400,
					Height = 150,
					WindowStartupLocation = WindowStartupLocation.CenterOwner
				};
				var panel = new StackPanel
				{
					Margin = new Thickness(10),
					Spacing = 10
				};
				panel.Children.Add(new TextBlock
				{
					Text = "Пожалуйста, выберите входной файл и выходную папку.",
					TextWrapping = TextWrapping.Wrap
				});
				var okButton = new Button
				{
					Content = "OK",
					HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
					Width = 80,
					Margin = new Thickness(0, 10, 0, 0)
				};
				okButton.Click += (_, _) => alert.Close();
				panel.Children.Add(okButton);
				alert.Content = panel;
				await alert.ShowDialog(this);
				return;
			}

			int idx = MetricComboBox.SelectedIndex;
			if (idx < 0 || idx >= calculators.Count)
				return;

			var calculator = calculators[idx];
			var data = GetDataLoader(inputPath).LoadData(inputPath);
			calculator.initialize(data);

			var infCalc = new InformativenessCalculatorDefault();
			var result = infCalc.Calculate(data, calculator);

			double[] xs = Enumerable.Range(1, result.informativenessList.Count)
									.Select(i => (double)i).ToArray();
			double[] ys = result.informativenessList.ToArray();

			var plotCtrl = this.FindControl<AvaPlot>("PlotControl");
			if (plotCtrl != null)
			{
				plotCtrl.Plot.Clear();
				plotCtrl.Plot.Title("Информативность признаков");
				plotCtrl.Plot.XLabel("Номер признака");
				plotCtrl.Plot.YLabel("Коэффициент информативности");

				plotCtrl.Plot.Add.Scatter(xs, ys);

				double xMax = xs.Max();
				double yMax = ys.Max();
				double xMin = xs.Min();
				double yMin = ys.Min();

				plotCtrl.Plot.Axes.SetLimits(xMin - 2, xMax + 2, yMin - 2, yMax + 2);

				plotCtrl.Refresh();
			}

			if (plotCtrl != null)
			{
				string metricName = calculator.GetType();
				string fileName = $"informativeness_{metricName}.png";
				string fullPath = Path.Combine(outputPath, fileName);
				plotCtrl.Plot.SavePng(fullPath, 1920, 1080);

				string dataFileName = $"informativeness_{metricName}.txt";
				string dataFilePath = Path.Combine(outputPath, dataFileName);

				try
				{
					File.WriteAllLines(dataFilePath, ys.Select(y => y.ToString("F2")));
				} catch (Exception ex)
				{
					var errorAlert = new Window
					{
						Title = "Ошибка",
						Width = 400,
						Height = 150,
						WindowStartupLocation = WindowStartupLocation.CenterOwner
					};
					var errorPanel = new StackPanel
					{
						Margin = new Thickness(10),
						Spacing = 10
					};
					errorPanel.Children.Add(new TextBlock
					{
						Text = $"Ошибка при сохранении данных: {ex.Message}",
						TextWrapping = TextWrapping.Wrap
					});
					var okButton = new Button
					{
						Content = "OK",
						HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center,
						Width = 80,
						Margin = new Thickness(0, 10, 0, 0)
					};
					okButton.Click += (_, _) => errorAlert.Close();
					errorPanel.Children.Add(okButton);
					errorAlert.Content = errorPanel;
					await errorAlert.ShowDialog(this);
				}

				var infoLabel = this.FindControl<TextBlock>("InfoLabel");
				if (infoLabel != null)
					infoLabel.Text = $"Image saved at: {fullPath}";

				var savedLabel = this.FindControl<TextBlock>("InformativenessSavedLabel");
				if (savedLabel != null)
				{
					savedLabel.Text = $"Informativeness values saved in: {dataFilePath}";
					savedLabel.IsVisible = true;
				}

			}
		}


		private IDataLoader GetDataLoader(string filePath) => new DataLoadTxt();
	}
}
