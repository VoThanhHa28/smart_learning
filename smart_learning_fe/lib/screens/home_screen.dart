import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:smart_learning/screens/query_screen.dart';
import 'package:smart_learning/screens/upload_screen.dart';
import '../../../services/course_service.dart';
import '../../../models/course.dart';
import 'dart:async';
import 'dart:io';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  final CourseService _courseService = CourseService();
  final TextEditingController _searchController = TextEditingController();

  List<Course> _courses = [];
  List<Course> _filteredCourses = [];
  bool _isLoading = true;
  String? _error;
  User? _currentUser;
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat();
    _checkUserAndLoadCourses();
  }

  @override
  void dispose() {
    _searchController.dispose();
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _checkUserAndLoadCourses() async {
    _currentUser = FirebaseAuth.instance.currentUser;

    print('');
    print('='*80);
    print('🏠 [HOME SCREEN] Initialization');
    print('='*80);

    if (_currentUser == null) {
      print('❌ ERROR: No user logged in!');
      print('='*80);

      setState(() {
        _isLoading = false;
        _error = 'Vui lòng đăng nhập để xem khóa học';
      });
      return;
    }

    print('✅ User authenticated:');
    print('   UID: ${_currentUser!.uid}');
    print('   Email: ${_currentUser!.email}');
    print('='*80);
    print('');

    await _loadCourses();
  }

  Future<void> _loadCourses() async {
    if (_currentUser == null) {
      setState(() {
        _isLoading = false;
        _error = 'User not logged in';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      print('📚 Loading courses for user: ${_currentUser!.uid}');

      final courses = await _courseService.getUserCourses(_currentUser!.uid);

      if (!mounted) return;

      setState(() {
        _courses = courses;
        _filteredCourses = courses;
        _isLoading = false;
      });

      print('✅ UI updated with ${courses.length} courses');
    } on TimeoutException catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Timeout: Không thể kết nối tới server';
        _isLoading = false;
      });
      print('❌ Timeout error: $e');
    } on SocketException catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Lỗi kết nối: Không thể kết nối tới server';
        _isLoading = false;
      });
      print('❌ Socket error: $e');
    } catch (e, stackTrace) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
      print('❌ Error: $e');
      print('Stack trace: $stackTrace');
    }
  }

  void _filterCourses(String query) {
    if (!mounted) return;

    setState(() {
      if (query.isEmpty) {
        _filteredCourses = _courses;
      } else {
        _filteredCourses = _courses.where((course) {
          return course.subject.toLowerCase().contains(query.toLowerCase()) ||
              course.courseId.toLowerCase().contains(query.toLowerCase());
        }).toList();
      }
    });

    print('🔎 Filtered: ${_filteredCourses.length}/${_courses.length} courses');
  }

  Future<void> _navigateToUpload() async {
    final result = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (context) => const UploadScreen(),
      ),
    );

    if (!mounted) return;

    if (result == true) {
      print('📚 Upload completed, reloading courses...');
      _loadCourses();
    }
  }

  Future<void> _navigateToCourse(Course course) async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => QueryScreen(
          subject: course.subject,
          courseId: course.courseId,
          docId: course.courseId,
        ),
      ),
    );
  }

  String _capitalize(String text) {
    if (text.isEmpty) return text;
    final normalized = text.replaceAll('_', ' ');
    return normalized.split(' ').map((word) {
      if (word.isEmpty) return word;
      return word[0].toUpperCase() + word.substring(1).toLowerCase();
    }).join(' ');
  }

  void _showUserMenu() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // User info
            Row(
              children: [
                CircleAvatar(
                  radius: 30,
                  backgroundColor: Colors.blue.shade100,
                  backgroundImage: _currentUser?.photoURL != null
                      ? NetworkImage(_currentUser!.photoURL!)
                      : null,
                  child: _currentUser?.photoURL == null
                      ? Icon(Icons.person, size: 30, color: Colors.blue.shade700)
                      : null,
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _currentUser?.displayName ?? 'User',
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _currentUser?.email ?? '',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const Divider(height: 32),

            // Stats
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatItem(
                  Icons.school,
                  '${_courses.length}',
                  'Khóa học',
                ),
                Container(width: 1, height: 40, color: Colors.grey[300]),
                _buildStatItem(
                  Icons.history_edu,
                  'Active',
                  'Trạng thái',
                ),
              ],
            ),

            const Divider(height: 32),

            // Actions
            ListTile(
              leading: const Icon(Icons.refresh, color: Colors.blue),
              title: const Text('Làm mới danh sách'),
              onTap: () {
                Navigator.pop(context);
                _loadCourses();
              },
            ),
            ListTile(
              leading: const Icon(Icons.logout, color: Colors.red),
              title: const Text('Đăng xuất'),
              onTap: () async {
                Navigator.pop(context);
                await FirebaseAuth.instance.signOut();
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(IconData icon, String value, String label) {
    return Column(
      children: [
        Icon(icon, color: Colors.blue.shade600),
        const SizedBox(height: 8),
        Text(
          value,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Smart Learning',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            if (_currentUser != null)
              Text(
                _currentUser!.email ?? 'No email',
                style: TextStyle(
                  fontSize: 11,
                  color: Colors.grey[600],
                  fontWeight: FontWeight.normal,
                ),
              ),
          ],
        ),
        actions: [
          if (_courses.isNotEmpty)
            IconButton(
              icon: Badge(
                label: Text('${_courses.length}'),
                child: const Icon(Icons.school_outlined),
              ),
              onPressed: () {},
              tooltip: '${_courses.length} khóa học',
            ),
          IconButton(
            icon: CircleAvatar(
              radius: 16,
              backgroundColor: Colors.blue.shade100,
              backgroundImage: _currentUser?.photoURL != null
                  ? NetworkImage(_currentUser!.photoURL!)
                  : null,
              child: _currentUser?.photoURL == null
                  ? Icon(Icons.person, size: 18, color: Colors.blue.shade700)
                  : null,
            ),
            onPressed: _showUserMenu,
            tooltip: 'Tài khoản',
          ),
          const SizedBox(width: 8),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(
            height: 1,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.blue.shade200,
                  Colors.purple.shade200,
                ],
              ),
            ),
          ),
        ),
      ),
      body: Column(
        children: [
          // Search bar
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.03),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(12),
              ),
              child: TextField(
                controller: _searchController,
                decoration: InputDecoration(
                  hintText: 'Tìm kiếm khóa học...',
                  hintStyle: TextStyle(color: Colors.grey[500]),
                  prefixIcon: Icon(Icons.search, color: Colors.grey[600]),
                  suffixIcon: _searchController.text.isNotEmpty
                      ? IconButton(
                    icon: const Icon(Icons.clear),
                    onPressed: () {
                      _searchController.clear();
                      _filterCourses('');
                    },
                  )
                      : null,
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 14,
                  ),
                ),
                onChanged: _filterCourses,
              ),
            ),
          ),

          // Course list
          Expanded(
            child: _buildBody(),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _navigateToUpload,
        icon: const Icon(Icons.add),
        label: const Text('Upload PDF'),
        backgroundColor: Colors.blue.shade600,
        elevation: 4,
      ),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            RotationTransition(
              turns: _animationController,
              child: Icon(
                Icons.school,
                size: 64,
                color: Colors.blue.shade300,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Đang tải khóa học...',
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[600],
              ),
            ),
          ],
        ),
      );
    }

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.error_outline,
                  size: 64,
                  color: Colors.red.shade400,
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'Oops!',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey[800],
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                ),
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: _loadCourses,
                icon: const Icon(Icons.refresh),
                label: const Text('Thử lại'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue.shade600,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (_courses.isEmpty) {
      return Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.blue.shade100, Colors.purple.shade100],
                  ),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.school_outlined,
                  size: 80,
                  color: Colors.blue.shade700,
                ),
              ),
              const SizedBox(height: 32),
              Text(
                'Chưa có khóa học nào',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey[800],
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Hãy bắt đầu bằng cách upload\ntài liệu học tập của bạn',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _navigateToUpload,
                icon: const Icon(Icons.upload_file),
                label: const Text('Upload PDF đầu tiên'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue.shade600,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (_filteredCourses.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.search_off,
              size: 80,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 24),
            Text(
              'Không tìm thấy khóa học',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.grey[800],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Thử tìm kiếm với từ khóa khác',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadCourses,
      color: Colors.blue.shade600,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _filteredCourses.length,
        itemBuilder: (context, index) {
          final course = _filteredCourses[index];
          return _buildCourseCard(course, index);
        },
      ),
    );
  }

  Widget _buildCourseCard(Course course, int index) {
    // Color palette based on index
    final colors = [
      [Colors.blue.shade500, Colors.blue.shade700],
      [Colors.purple.shade500, Colors.purple.shade700],
      [Colors.orange.shade500, Colors.orange.shade700],
      [Colors.green.shade500, Colors.green.shade700],
      [Colors.pink.shade500, Colors.pink.shade700],
    ];

    final colorPair = colors[index % colors.length];

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: colorPair,
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: colorPair[0].withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => _navigateToCourse(course),
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                // Icon
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.book,
                    color: Colors.white,
                    size: 32,
                  ),
                ),
                const SizedBox(width: 16),

                // Content
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _capitalize(course.subject),
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        course.courseId,
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.white.withOpacity(0.9),
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),

                // Arrow
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.arrow_forward_ios,
                    color: Colors.white,
                    size: 16,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}