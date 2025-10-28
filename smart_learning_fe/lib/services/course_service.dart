import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:smart_learning/core/api_config.dart';
import '../models/course.dart';


class CourseService {
  static String get baseUrl => ApiConfig.baseUrl;

  /// Get courses by user_id
  Future<List<Course>> getUserCourses(String userId, {String? search}) async {
    try {
      print('');
      print('='*80);
      print('🔍 [COURSE SERVICE] GET USER COURSES');
      print('='*80);
      print('👤 User ID: $userId');
      if (search != null && search.isNotEmpty) {
        print('🔎 Search: $search');
      }

      // ✅ Build URL với query parameter
      final uri = Uri.parse('$baseUrl/index/my_courses').replace(
        queryParameters: {'user_id': userId},
      );

      print('📍 URL: $uri');
      print('⏳ Sending request...');
      print('='*80);

      final response = await http.get(
        uri,
        headers: {'Accept': 'application/json'},
      ).timeout(const Duration(seconds: 30));

      print('');
      print('📥 Response:');
      print('   Status: ${response.statusCode}');
      print('   Body: ${response.body}');
      print('');

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);
        final List<dynamic> coursesJson = data['courses'] as List<dynamic>;

        print('✅ API returned ${coursesJson.length} courses');

        List<Course> courses = coursesJson
            .map((e) => Course.fromJson(e as Map<String, dynamic>))
            .toList();

        // ✅ Client-side filtering (nếu có search)
        if (search != null && search.isNotEmpty) {
          final searchLower = search.toLowerCase();
          final originalCount = courses.length;

          courses = courses.where((course) {
            return course.subject.toLowerCase().contains(searchLower) ||
                course.courseId.toLowerCase().contains(searchLower);
          }).toList();

          print('🔎 Filtered from $originalCount to ${courses.length} courses');
        }

        print('✅ Returning ${courses.length} courses');
        print('='*80);
        print('');

        return courses;
      } else {
        print('❌ Error: Status ${response.statusCode}');
        print('   Body: ${response.body}');
        print('='*80);
        print('');

        throw Exception('Failed to load courses (${response.statusCode}): ${response.body}');
      }
    } on TimeoutException catch (e) {
      print('');
      print('❌ TIMEOUT ERROR:');
      print('   $e');
      print('='*80);
      print('');

      throw Exception('Timeout: Không thể kết nối tới server.\nVui lòng kiểm tra backend có đang chạy không?');
    } on SocketException catch (e) {
      print('');
      print('❌ NETWORK ERROR:');
      print('   $e');
      print('   URL: $baseUrl/index/my_courses');
      print('='*80);
      print('');

      throw Exception('Lỗi kết nối: Không thể kết nối tới server.\nBackend có đang chạy không?');
    } catch (e, stackTrace) {
      print('');
      print('❌ UNEXPECTED ERROR:');
      print('   Error: $e');
      print('   Stack:');
      print('$stackTrace');
      print('='*80);
      print('');

      rethrow;
    }
  }
}