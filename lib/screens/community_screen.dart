
// lib/screens/community_screen.dart (New Screen)
import 'package:flutter/material.dart';
import 'package:my_health_app/utils/app_constants.dart';

class CommunityScreen extends StatelessWidget {
  const CommunityScreen({super.key});

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
        title: const Text('Community'),
        backgroundColor: AppConstants.primaryColor,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.search),
            onPressed: () {
              // TODO: Implement search
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                _CommunityPostCard(
                  profileImage: 'https://placehold.co/50x50/E0E0E0/000000?text=P',
                  username: 'Preeti',
                  date: '06/06/25',
                  postTitle: 'Battu Mango Lassi',
                  postContent: 'A power packed health drink full of protein, vitamin A,C , fibre, h ...',
                  postImage: 'https://placehold.co/300x200/E0E0E0/000000?text=Mango+Lassi',
                  likes: 2,
                  comments: 1,
                  timeAgo: '1 Days ago',
                ),
                _CommunityPostCard(
                  profileImage: 'https://placehold.co/50x50/E0E0E0/000000?text=S',
                  username: 'Suchitra Tiwari',
                  date: '06/06/25',
                  postTitle: 'Thyroid problem in india',
                  postContent: 'In India, around 60 % of people with thyroid disorders are...',
                  postImage: null, // No image for this post
                  likes: 0,
                  comments: 0,
                  timeAgo: '1 Days ago',
                ),
              ],
            ),
          ),
          Positioned(
            bottom: 20,
            right: 20,
            child: FloatingActionButton(
              onPressed: () {
                // TODO: Create new post
              },
              backgroundColor: AppConstants.primaryColor,
              foregroundColor: Colors.white,
              child: const Icon(Icons.add),
            ),
          ),
        ],
      ),
    );
  }
}

class _CommunityPostCard extends StatelessWidget {
  final String profileImage;
  final String username;
  final String date;
  final String postTitle;
  final String postContent;
  final String? postImage;
  final int likes;
  final int comments;
  final String timeAgo;

  const _CommunityPostCard({
    required this.profileImage,
    required this.username,
    required this.date,
    required this.postTitle,
    required this.postContent,
    this.postImage,
    required this.likes,
    required this.comments,
    required this.timeAgo,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 20,
                  backgroundImage: NetworkImage(profileImage),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        username,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                          color: AppConstants.textColor,
                        ),
                      ),
                      Text(
                        date,
                        style: const TextStyle(
                          fontSize: 12,
                          color: AppConstants.greyColor,
                        ),
                      ),
                    ],
                  ),
                ),
                TextButton(
                  onPressed: () {
                    // TODO: Implement follow logic
                  },
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.zero, // Remove padding
                    minimumSize: Size.zero, // Remove minimum size constraints
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap, // Shrink tap target
                  ),
                  child: const Text(
                    'Follow',
                    style: TextStyle(color: AppConstants.primaryColor, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(width: 8),
                const Icon(Icons.more_vert, color: AppConstants.greyColor),
              ],
            ),
            const SizedBox(height: 15),
            Text(
              postTitle,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
                color: AppConstants.textColor,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              postContent,
              style: const TextStyle(
                fontSize: 14,
                color: AppConstants.textColor,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            if (postImage != null) ...[
              const SizedBox(height: 15),
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.network(
                  postImage!,
                  width: double.infinity,
                  height: 200,
                  fit: BoxFit.cover,
                ),
              ),
            ],
            const SizedBox(height: 15),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.favorite_border, size: 20, color: AppConstants.greyColor),
                    const SizedBox(width: 4),
                    Text('$likes', style: const TextStyle(color: AppConstants.greyColor)),
                    const SizedBox(width: 16),
                    const Icon(Icons.comment_outlined, size: 20, color: AppConstants.greyColor),
                    const SizedBox(width: 4),
                    Text('$comments', style: const TextStyle(color: AppConstants.greyColor)),
                  ],
                ),
                Text(
                  timeAgo,
                  style: const TextStyle(fontSize: 12, color: AppConstants.greyColor),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}