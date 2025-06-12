
// lib/screens/meal_tracker_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';
import 'package:charts_flutter/flutter.dart' as charts; // Requires adding charts_flutter dependency

class MealTrackerScreen extends StatelessWidget {
  const MealTrackerScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        title: const Text('Meal Tracker'),
        backgroundColor: Colors.transparent, // Match the original design
        foregroundColor: AppConstants.textColor,
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_none),
            onPressed: () {
              // TODO: Handle notifications
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: TextField(
                    decoration: InputDecoration(
                      hintText: 'Search for food items ...',
                      prefixIcon: const Icon(Icons.search, color: AppConstants.greyColor),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12.0),
                        borderSide: BorderSide.none,
                      ),
                      filled: true,
                      fillColor: AppConstants.lightGrey,
                      contentPadding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 20.0),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                GestureDetector(
                  onTap: () {
                    // TODO: Implement scan functionality
                  },
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppConstants.lightGrey,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Column(
                      children: [
                        Icon(Icons.qr_code_scanner, color: AppConstants.primaryColor, size: 24),
                        Text('Scan', style: TextStyle(fontSize: 12, color: AppConstants.primaryColor)),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 30),
            Text(
              "Today's Meals",
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            _MealEntryCard(
              image: 'https://placehold.co/80x80/E0E0E0/000000?text=Salmon',
              title: 'Grilled Salmon & Asparagus',
              time: '1:30 PM',
              macros: '35g Prot 15g Carb 28g Fats',
              calories: '450 kcal',
            ),
            _MealEntryCard(
              image: 'https://placehold.co/80x80/E0E0E0/000000?text=Yogurt',
              title: 'Greek Yogurt with Berries',
              time: '10:00 AM',
              macros: '20g Prot 25g Carbs 2g Fats',
              calories: '180 kcal',
            ),
            _MealEntryCard(
              image: 'https://placehold.co/80x80/E0E0E0/000000?text=StirFry',
              title: 'Chicken & Veggie Stir-fry',
              time: '7:00 PM',
              macros: '40g Prot 30g Carb 25g Fats',
              calories: '550 kcal',
            ),
            const SizedBox(height: 30),
            Text(
              'Daily Nutritional Breakdown',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.1),
                    spreadRadius: 1,
                    blurRadius: 5,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        '1180',
                        style: Theme.of(context).textTheme.displayLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: AppConstants.primaryColor,
                              fontSize: 48,
                            ),
                      ),
                      Text(
                        ' kcal of 2000 kcal',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              color: AppConstants.textColor,
                            ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _MacroSummary(label: 'Protein', value: '95g'),
                      _MacroSummary(label: 'Carbs', value: '70g'),
                      _MacroSummary(label: 'Fats', value: '55g'),
                    ],
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                    height: 200,
                    child: _buildNutritionalChart(), // Bar chart for nutritional breakdown
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNutritionalChart() {
    final data = [
      NutrientSeries('Protein', 95, AppConstants.primaryColor),
      NutrientSeries('Carbs', 70, AppConstants.accentColor),
      NutrientSeries('Fats', 55, AppConstants.darkBlue),
    ];

    List<charts.Series<NutrientSeries, String>> series = [
      charts.Series(
        id: 'Nutrients',
        data: data,
        domainFn: (NutrientSeries series, _) => series.category,
        measureFn: (NutrientSeries series, _) => series.value,
        colorFn: (NutrientSeries series, _) =>
            charts.ColorUtil.fromDartColor(series.color),
        // Set a fixed width for the bars
        fillPatternFn: (_, __) => charts.FillPatternType.solid,
        displayName: 'Nutrients',
      )
    ];

    return charts.BarChart(
      series,
      animate: true,
      domainAxis: charts.OrdinalAxisSpec(
        renderSpec: charts.SmallTickRendererSpec(
          labelStyle: charts.TextStyleSpec(
            color: charts.ColorUtil.fromDartColor(AppConstants.textColor),
          ),
          lineStyle: charts.LineStyleSpec(
            color: charts.ColorUtil.fromDartColor(AppConstants.greyColor),
          ),
        ),
      ),
      primaryMeasureAxis: charts.NumericAxisSpec(
        renderSpec: charts.SmallTickRendererSpec(
          labelStyle: charts.TextStyleSpec(
            color: charts.ColorUtil.fromDartColor(AppConstants.greyColor),
          ),
          lineStyle: charts.LineStyleSpec(
            color: charts.ColorUtil.fromDartColor(AppConstants.greyColor),
          ),
        ),
        tickProviderSpec: const charts.BasicNumericTickProviderSpec(
          desiredTickCount: 5, // Example: 0, 25, 50, 75, 100
        ),
      ),
      barRendererDecorator: charts.BarLabelDecorator<String>(),
      // Hide the legend, as the labels are directly on the bars or below.
      defaultRenderer: charts.BarRendererConfig(
        groupingType: charts.BarGroupingType.grouped,
        cornerStrategy: const charts.ConstCornerStrategy(30), // Rounded corners for bars
      ),
      // No external library for charts_flutter, it needs to be added to pubspec.yaml
      // charts_flutter: ^0.12.0
    );
  }
}

class NutrientSeries {
  final String category;
  final int value;
  final Color color;

  NutrientSeries(this.category, this.value, this.color);
}


class _MealEntryCard extends StatelessWidget {
  final String image;
  final String title;
  final String time;
  final String macros;
  final String calories;

  const _MealEntryCard({
    required this.image,
    required this.title,
    required this.time,
    required this.macros,
    required this.calories,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                image,
                height: 70,
                width: 70,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: AppConstants.textColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    time,
                    style: const TextStyle(
                      fontSize: 14,
                      color: AppConstants.greyColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    macros,
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppConstants.greyColor,
                    ),
                  ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  calories,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppConstants.primaryColor,
                  ),
                ),
                const SizedBox(height: 8),
                const Icon(Icons.edit, size: 18, color: AppConstants.greyColor),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MacroSummary extends StatelessWidget {
  final String label;
  final String value;

  const _MacroSummary({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: AppConstants.textColor,
          ),
        ),
        Text(
          label,
          style: const TextStyle(
            fontSize: 14,
            color: AppConstants.greyColor,
          ),
        ),
      ],
    );
  }
}