
// lib/screens/fitness_sync_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';
import 'package:charts_flutter/flutter.dart' as charts; // Requires adding charts_flutter dependency

class FitnessSyncScreen extends StatelessWidget {
  const FitnessSyncScreen({super.key});

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
        title: const Text('Fitness Sync'),
        backgroundColor: Colors.transparent,
        foregroundColor: AppConstants.textColor,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'You have achieved 80 % of your goal today',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: AppConstants.textColor,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 20),
            Center(
              child: Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 180,
                    height: 180,
                    child: CircularProgressIndicator(
                      value: 0.8, // 80% progress
                      strokeWidth: 10,
                      backgroundColor: AppConstants.lightGrey,
                      color: AppConstants.primaryColor,
                    ),
                  ),
                  Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.directions_walk, size: 40, color: AppConstants.textColor),
                      const SizedBox(height: 8),
                      Text(
                        '11,857',
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: AppConstants.textColor,
                        ),
                      ),
                      Text(
                        'Steps out of 18,000 steps',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: AppConstants.greyColor,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: const [
                _FitnessMetricCircle(
                  icon: Icons.local_fire_department,
                  value: '850',
                  unit: 'kcal',
                  label: 'Calories Burned',
                  iconColor: Colors.orange,
                ),
                _FitnessMetricCircle(
                  icon: Icons.map,
                  value: '5',
                  unit: 'km',
                  label: 'Distance Covered',
                  iconColor: Colors.blue,
                ),
                _FitnessMetricCircle(
                  icon: Icons.access_time,
                  value: '120',
                  unit: 'min',
                  label: 'Active Minutes',
                  iconColor: Colors.green,
                ),
              ],
            ),
            const SizedBox(height: 30),
            Text(
              'Activity Trend',
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
                      _TrendToggle('Today', true),
                      _TrendToggle('Weekly', false),
                      _TrendToggle('Monthly', false),
                    ],
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                    height: 200,
                    child: _buildActivityTrendChart(),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            Text(
              'Connected Devices',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 15),
            _ConnectedDeviceTile(
              icon: 'https://img.icons8.com/color/48/000000/garmin.png',
              deviceName: 'Garmin Forerunner 945',
              status: 'Connected',
              statusColor: AppConstants.primaryColor,
              isConnected: true,
            ),
            _ConnectedDeviceTile(
              icon: 'https://img.icons8.com/color/48/000000/apple-watch.png',
              deviceName: 'Apple Watch Series 7',
              status: 'Disconnected',
              statusColor: Colors.red,
              isConnected: false,
            ),
            _ConnectedDeviceTile(
              icon: 'https://img.icons8.com/color/48/000000/fitbit.png',
              deviceName: 'Fitbit Charge 5',
              status: 'Connected',
              statusColor: AppConstants.primaryColor,
              isConnected: true,
            ),
            const SizedBox(height: 30),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppConstants.primaryColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  const Icon(Icons.flash_on, color: AppConstants.primaryColor, size: 36),
                  const SizedBox(height: 10),
                  Text(
                    'Integrate Workouts',
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: AppConstants.textColor,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 5),
                  Text(
                    'Sync your workout data with your diet plan for optimized results.',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppConstants.greyColor,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {
                        // TODO: Connect new device logic
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppConstants.primaryColor,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16.0),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12.0),
                        ),
                      ),
                      child: const Text(
                        'Connect New Device',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActivityTrendChart() {
    final data = [
      ActivityData('Tue', 20),
      ActivityData('Wed', 40),
      ActivityData('Thu', 60),
      ActivityData('Fri', 80),
      ActivityData('Sat', 50),
      ActivityData('Sun', 30),
      ActivityData('Mon', 70), // Example data
    ];

    List<charts.Series<ActivityData, String>> series = [
      charts.Series(
        id: 'Activity Trend',
        data: data,
        domainFn: (ActivityData activity, _) => activity.day,
        measureFn: (ActivityData activity, _) => activity.value,
        colorFn: (_, __) => charts.ColorUtil.fromDartColor(AppConstants.primaryColor),
        areaColorFn: (_, __) => charts.ColorUtil.fromDartColor(AppConstants.primaryColor.withOpacity(0.2)),
        // Fill area under the line
      )
    ];

    return charts.LineChart(
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
      ),
      // No external library for charts_flutter, it needs to be added to pubspec.yaml
    );
  }
}

class ActivityData {
  final String day;
  final int value;

  ActivityData(this.day, this.value);
}

class _TrendToggle extends StatelessWidget {
  final String label;
  final bool isSelected;

  const _TrendToggle(this.label, this.isSelected);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8.0),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        decoration: BoxDecoration(
          color: isSelected ? AppConstants.primaryColor : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : AppConstants.textColor,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

class _FitnessMetricCircle extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final String value;
  final String unit;
  final String label;

  const _FitnessMetricCircle({
    required this.icon,
    required this.iconColor,
    required this.value,
    required this.unit,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: AppConstants.lightGrey, width: 2),
          ),
          child: Icon(icon, size: 30, color: iconColor),
        ),
        const SizedBox(height: 8),
        Text(
          '$value $unit',
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 16,
            color: AppConstants.textColor,
          ),
        ),
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: AppConstants.greyColor,
          ),
        ),
      ],
    );
  }
}

class _ConnectedDeviceTile extends StatelessWidget {
  final String icon;
  final String deviceName;
  final String status;
  final Color statusColor;
  final bool isConnected;

  const _ConnectedDeviceTile({
    required this.icon,
    required this.deviceName,
    required this.status,
    required this.statusColor,
    required this.isConnected,
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
          children: [
            Image.network(icon, height: 40, width: 40),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    deviceName,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      color: AppConstants.textColor,
                    ),
                  ),
                  Text(
                    status,
                    style: TextStyle(
                      fontSize: 14,
                      color: statusColor,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            if (isConnected)
              const Icon(Icons.wifi, color: AppConstants.primaryColor)
            else
              const Icon(Icons.wifi_off, color: AppConstants.greyColor),
          ],
        ),
      ),
    );
  }
}