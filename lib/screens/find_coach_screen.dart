
// lib/screens/find_coach_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class FindCoachScreen extends StatelessWidget {
  const FindCoachScreen({super.key});

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
        title: const Text('Find a Coach'),
        backgroundColor: AppConstants.primaryColor,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: TextField(
                    decoration: InputDecoration(
                      hintText: 'Search',
                      prefixIcon: const Icon(Icons.search, color: AppConstants.greyColor),
                      suffixIcon: IconButton(
                        icon: const Icon(Icons.mic, color: AppConstants.primaryColor),
                        onPressed: () {
                          // TODO: Implement voice search
                        },
                      ),
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
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppConstants.lightGrey,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.filter_list, color: AppConstants.primaryColor),
                ),
              ],
            ),
            const SizedBox(height: 20),
            SizedBox(
              height: 40, // Height for the filter chips
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: [
                  _FilterChip('Yoga', true),
                  _FilterChip('Sports Conditioning', false),
                  _FilterChip('Functional Training', false),
                  _FilterChip('Weight Management', false),
                ],
              ),
            ),
            const SizedBox(height: 20),
            GridView.count(
              crossAxisCount: 2,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              mainAxisSpacing: 16.0,
              crossAxisSpacing: 16.0,
              children: [
                _CoachCard(
                  image: 'https://placehold.co/100x100/E0E0E0/000000?text=Coach1',
                  name: 'Dt Jahnvi Jain',
                  experience: '2 years 0 mont...',
                  specialties: 'Spl: Yoga, Fitness...',
                  rating: '4.8',
                  reviews: '90',
                  price: '20 per minute',
                ),
                _CoachCard(
                  image: 'https://placehold.co/100x100/E0E0E0/000000?text=Coach2',
                  name: 'Dt. Neelu Yad...',
                  experience: '2 years 7 months',
                  specialties: 'Spl: Weight Mana...',
                  rating: '5.0',
                  reviews: '1',
                  price: '₹0 for', // Example free
                ),
                _CoachCard(
                  image: 'https://placehold.co/100x100/E0E0E0/000000?text=Coach3',
                  name: 'Palak Mittal',
                  experience: '12 years 1 mont...',
                  specialties: 'Spl: Weight Mana...',
                  rating: '5.0',
                  reviews: '1',
                  price: '₹45 per minute',
                ),
                _CoachCard(
                  image: 'https://placehold.co/100x100/E0E0E0/000000?text=Coach4',
                  name: 'Coach Bidus...',
                  experience: '5 years 3 mont...',
                  specialties: 'Spl: Sports Conditi...',
                  rating: '4.5',
                  reviews: '25',
                  price: '₹30 per minute',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final bool isSelected;

  const _FilterChip(this.label, this.isSelected);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8.0),
      child: Chip(
        label: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : AppConstants.textColor,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
        backgroundColor: isSelected ? AppConstants.primaryColor : AppConstants.lightGrey,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        side: BorderSide.none,
      ),
    );
  }
}

class _CoachCard extends StatelessWidget {
  final String image;
  final String name;
  final String experience;
  final String specialties;
  final String rating;
  final String reviews;
  final String price;

  const _CoachCard({
    required this.image,
    required this.name,
    required this.experience,
    required this.specialties,
    required this.rating,
    required this.reviews,
    required this.price,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.network(
                  image,
                  height: 80,
                  width: 80,
                  fit: BoxFit.cover,
                ),
              ),
            ),
            const SizedBox(height: 10),
            Text(
              name,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: AppConstants.textColor,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            Text(
              experience,
              style: const TextStyle(
                fontSize: 12,
                color: AppConstants.greyColor,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            Text(
              specialties,
              style: const TextStyle(
                fontSize: 12,
                color: AppConstants.greyColor,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.star, color: Colors.amber, size: 16),
                Text(
                  '$rating ($reviews) Reviews',
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppConstants.greyColor,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              price,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: AppConstants.primaryColor,
              ),
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  // TODO: Book coach logic
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppConstants.primaryColor,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 8.0),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8.0),
                  ),
                ),
                child: const Text(
                  'Book Now',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

