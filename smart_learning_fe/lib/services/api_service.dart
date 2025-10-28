import 'dart:convert';
import 'dart:io';
import 'dart:async';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:file_picker/file_picker.dart';
import 'package:smart_learning/core/api_config.dart';
import '../models/upload_response.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

class ApiService {
  static String get baseUrl => ApiConfig.baseUrl; // ✅ Dùng config

  /// Upload PDF file với user_id
  Future<UploadResponse> uploadPdf({
    required PlatformFile platformFile,
    required String subject,
    required String userId,
    String? courseId,
  }) async {
    try {
      print('🚀 Starting upload...');
      print('📝 File: ${platformFile.name} (${platformFile.size} bytes)');
      print('📝 Subject: $subject');
      print('📝 Course ID: ${courseId ?? "auto-generated"}');
      print('👤 User ID: $userId');
      print('📝 Target URL: $baseUrl/index/upload');

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/index/upload'),
      );

      // Add form fields
      request.fields['subject'] = subject;
      request.fields['user_id'] = userId;

      if (courseId != null && courseId.isNotEmpty) {
        request.fields['course_id'] = courseId;
      }

      // Add file
      http.MultipartFile multipartFile;

      if (kIsWeb) {
        if (platformFile.bytes == null) {
          throw Exception('File bytes are null. Cannot upload on web.');
        }

        multipartFile = http.MultipartFile.fromBytes(
          'file',
          platformFile.bytes!,
          filename: platformFile.name,
          contentType: MediaType('application', 'pdf'),
        );
        print('📱 Web upload - Size: ${platformFile.bytes!.length} bytes');
      } else {
        if (platformFile.path == null) {
          throw Exception('File path is null. Cannot upload on mobile.');
        }

        var file = File(platformFile.path!);
        if (!await file.exists()) {
          throw Exception('File does not exist at path: ${platformFile.path}');
        }

        var stream = http.ByteStream(file.openRead());
        var length = await file.length();

        multipartFile = http.MultipartFile(
          'file',
          stream,
          length,
          filename: platformFile.name,
          contentType: MediaType('application', 'pdf'),
        );
        print('📱 Mobile upload - Size: $length bytes');
      }

      request.files.add(multipartFile);

      // Send request
      print('⏳ Sending request...');
      var streamedResponse = await request.send().timeout(
        ApiConfig.uploadTimeout, // ✅ Dùng timeout từ config
        onTimeout: () {
          throw TimeoutException(
              'Upload timeout after ${ApiConfig.uploadTimeout.inSeconds} seconds.\n'
                  'File might be too large or backend is processing slowly.'
          );
        },
      );

      var responseBody = await streamedResponse.stream.bytesToString();

      print('📥 Response status: ${streamedResponse.statusCode}');
      print('📥 Response body: $responseBody');

      if (streamedResponse.statusCode == 200) {
        try {
          var jsonData = json.decode(responseBody);
          print('✅ JSON parsed successfully');

          if (jsonData is! Map<String, dynamic>) {
            throw Exception('Invalid response format: expected JSON object');
          }

          return UploadResponse.fromJson(jsonData);
        } catch (e) {
          print('❌ JSON parse error: $e');
          throw Exception('Failed to parse response: $e\nRaw: $responseBody');
        }
      } else if (streamedResponse.statusCode == 400) {
        try {
          var error = json.decode(responseBody);
          throw Exception(error['detail'] ?? responseBody);
        } catch (_) {
          throw Exception('Bad request: $responseBody');
        }
      } else {
        throw Exception(
            'Upload failed with status ${streamedResponse.statusCode}:\n$responseBody'
        );
      }
    } on SocketException catch (e) {
      print('❌ Network error: $e');
      throw Exception(
          'Không thể kết nối tới backend tại $baseUrl\n'
              'Kiểm tra:\n'
              '1. Backend có đang chạy không?\n'
              '2. Địa chỉ: $baseUrl\n'
              '3. Firewall có chặn không?'
      );
    } on TimeoutException catch (e) {
      print('❌ Timeout error: $e');
      throw Exception('Request timeout. File có thể quá lớn hoặc backend xử lý chậm.');
    } catch (e) {
      print('❌ Upload error: $e');
      rethrow;
    }
  }
}