

// lib/screens/diet_plan_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class DietPlanScreen extends StatelessWidget {
  const DietPlanScreen({super.key});

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
        title: const Text('Diet Plan'),
        backgroundColor: AppConstants.primaryColor,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              color: AppConstants.primaryColor,
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
              child: Column(
                children: [
                  // Month Navigator
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      IconButton(onPressed: () {}, icon: const Icon(Icons.arrow_back_ios, color: Colors.white, size: 16)),
                      const Text(
                        'MARCH 2025',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      IconButton(onPressed: () {}, icon: const Icon(Icons.arrow_forward_ios, color: Colors.white, size: 16)),
                    ],
                  ),
                  const SizedBox(height: 10),
                  // Day Selector
                  SizedBox(
                    height: 70, // Height for day selection
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: 7, // Example days
                      itemBuilder: (context, index) {
                        final dayOfWeek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][index % 7];
                        final date = 3 + index; // Example dates
                        final isSelected = index == 1; // 'Tue 4' is selected in the image

                        return GestureDetector(
                          onTap: () {
                            // TODO: Handle day selection
                          },
                          child: Container(
                            width: 60,
                            margin: const EdgeInsets.symmetric(horizontal: 4),
                            decoration: BoxDecoration(
                              color: isSelected ? AppConstants.accentColor : Colors.transparent,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  dayOfWeek,
                                  style: TextStyle(
                                    color: isSelected ? Colors.white : AppConstants.lightGrey,
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  '$date',
                                  style: TextStyle(
                                    color: isSelected ? Colors.white : Colors.white,
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 10),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.fastfood_outlined, color: AppConstants.primaryColor),
                      const SizedBox(width: 8),
                      Text(
                        'Meals',
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: AppConstants.textColor,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 15),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        // TODO: Navigate to Shopping List
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppConstants.primaryColor,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16.0),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12.0),
                        ),
                      ),
                      icon: const Icon(Icons.shopping_bag_outlined),
                      label: const Text(
                        'Shopping List',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  _MealCard(
                    image: 'https://placehold.co/100x100/E0E0E0/000000?text=Smoothie',
                    title: 'Peanut Butter Cup Smoothie',
                    mealType: 'Breakfast',
                    prepTime: '5 MIN',
                  ),
                  _MealCard(
                    image: 'https://placehold.co/100x100/E0E0E0/000000?text=Salad',
                    title: 'Grab & Go Broccoli Quinoa Salad',
                    mealType: 'Lunch',
                    prepTime: '10 MIN',
                  ),
                  _MealCard(
                    image: 'https://placehold.co/100x100/E0E0E0/000000?text=Smoothie2',
                    title: 'Choc Caramel Super Smoothie',
                    mealType: 'Post-Workout Meal',
                    prepTime: '5 MIN',
                  ),
                  _MealCard(
                    image: 'https://placehold.co/100x100/E0E0E0/000000?text=Spaghetti',
                    title: 'Weeknight Pesto & Ricotta Spaghetti',
                    mealType: 'Dinner',
                    prepTime: '15 MIN',
                  ),
                  _MealCard(
                    image: 'https://placehold.co/100x100/E0E0E0/000000?text=Yogurt',
                    title: 'Berry Granola Yogurt Bowl',
                    mealType: 'Snack',
                    prepTime: '3 MIN',
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MealCard extends StatelessWidget {
  final String image;
  final String title;
  final String mealType;
  final String prepTime;

  const _MealCard({
    required this.image,
    required this.title,
    required this.mealType,
    required this.prepTime,
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
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                image,
                height: 80,
                width: 80,
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
                    mealType,
                    style: const TextStyle(
                      fontSize: 14,
                      color: AppConstants.greyColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      const Icon(Icons.timer_outlined, size: 16, color: AppConstants.greyColor),
                      const SizedBox(width: 4),
                      Text(
                        prepTime,
                        style: const TextStyle(
                          fontSize: 12,
                          color: AppConstants.greyColor,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            IconButton(
              icon: const Icon(Icons.swap_horiz, color: AppConstants.primaryColor),
              onPressed: () {
                // TODO: Implement meal swap
              },
            ),
          ],
        ),
      ),
    );
  }
}
