import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:smart_learning/core/api_config.dart';
import '../models/query_models.dart';

class StreamingApiService {
  static String get baseUrl => ApiConfig.baseUrl;

  /// Query với streaming response
  Stream<String> queryStream(QueryRequest request) async* {
    try {
      // ✅ Lấy user từ Firebase
      final user = FirebaseAuth.instance.currentUser;

      print('');
      print('='*80);
      print('🔍 [STREAMING API] QUERY REQUEST');
      print('='*80);

      if (user == null) {
        print('❌ ERROR: User not logged in!');
        print('='*80);
        throw Exception('User not logged in. Please login first.');
      }

      print('👤 Firebase User:');
      print('   UID: ${user.uid}');
      print('   Email: ${user.email}');
      print('   Display Name: ${user.displayName}');
      print('');
      print('📝 Request Details:');
      print('   Question: ${request.question}');
      print('   Subject: ${request.subject}');
      print('   Course ID: ${request.courseId}');
      print('   Top K: ${request.topK}');
      print('');
      print('🌐 API Details:');
      print('   Base URL: $baseUrl');
      print('   Endpoint: $baseUrl/query');

      final client = http.Client();

      final streamRequest = http.Request(
        'POST',
        Uri.parse('$baseUrl/query'),
      );

      // ✅ Headers
      final headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/plain, text/event-stream, application/json',
        'X-User-ID': user.uid,
      };

      streamRequest.headers.addAll(headers);

      // ✅ Body
      final bodyMap = request.toJson();
      final bodyJson = json.encode(bodyMap);
      streamRequest.body = bodyJson;

      print('');
      print('📋 Request Headers:');
      headers.forEach((key, value) {
        print('   $key: $value');
      });
      print('');
      print('📦 Request Body:');
      print('   Raw JSON: $bodyJson');
      print('   Parsed: $bodyMap');
      print('');
      print('⏳ Sending streaming request...');
      print('='*80);

      final streamedResponse = await client.send(streamRequest).timeout(
        const Duration(seconds: 60),
        onTimeout: () {
          print('');
          print('❌ TIMEOUT: Request took more than 60 seconds');
          print('='*80);
          throw TimeoutException(
              'Query timeout after 60 seconds.\n'
                  'Backend might be processing or LLM is slow.'
          );
        },
      );

      print('');
      print('📥 Response Received:');
      print('   Status Code: ${streamedResponse.statusCode}');
      print('   Reason: ${streamedResponse.reasonPhrase}');
      print('   Headers:');
      streamedResponse.headers.forEach((key, value) {
        print('     $key: $value');
      });
      print('');

      if (streamedResponse.statusCode != 200) {
        final errorBody = await streamedResponse.stream.bytesToString();

        print('❌ ERROR RESPONSE:');
        print('   Status: ${streamedResponse.statusCode}');
        print('   Body: $errorBody');
        print('='*80);

        // Try parse JSON error
        try {
          final errorJson = json.decode(errorBody);
          final errorDetail = errorJson['detail'] ?? errorBody;
          throw Exception('Query failed (${streamedResponse.statusCode}): $errorDetail');
        } catch (e) {
          throw Exception('Query failed (${streamedResponse.statusCode}): $errorBody');
        }
      }

      // ✅ Stream chunks
      print('📤 Streaming Response Chunks:');
      print('-'*80);

      int chunkCount = 0;
      int totalBytes = 0;
      final startTime = DateTime.now();

      await for (var chunk in streamedResponse.stream.transform(utf8.decoder)) {
        if (chunk.isNotEmpty) {
          chunkCount++;
          totalBytes += chunk.length;

          // Log every 10 chunks
          if (chunkCount % 10 == 0) {
            final elapsed = DateTime.now().difference(startTime);
            print('   📦 Chunk $chunkCount: $totalBytes bytes (${elapsed.inSeconds}s)');
          }

          yield chunk;
        }
      }

      final totalTime = DateTime.now().difference(startTime);

      print('-'*80);
      print('✅ Streaming Completed Successfully!');
      print('   Total Chunks: $chunkCount');
      print('   Total Bytes: $totalBytes');
      print('   Total Time: ${totalTime.inSeconds}s (${totalTime.inMilliseconds}ms)');
      print('='*80);
      print('');

      client.close();

    } on TimeoutException catch (e) {
      print('');
      print('❌ TIMEOUT ERROR:');
      print('   $e');
      print('='*80);
      print('');
      rethrow;
    } on SocketException catch (e) {
      print('');
      print('❌ NETWORK ERROR:');
      print('   $e');
      print('   URL: $baseUrl/query');
      print('');
      print('💡 Possible causes:');
      print('   1. Backend is not running');
      print('   2. Wrong URL/port');
      print('   3. Firewall blocking');
      print('   4. Network connectivity issue');
      print('='*80);
      print('');
      throw Exception(
          'Không thể kết nối tới backend.\n'
              'URL: $baseUrl/query\n'
              'Error: $e'
      );
    } catch (e, stackTrace) {
      print('');
      print('❌ UNEXPECTED ERROR:');
      print('   Error: $e');
      print('   Stack Trace:');
      print('$stackTrace');
      print('='*80);
      print('');
      rethrow;
    }
  }

  /// Query complete (non-streaming) - For testing
  Future<Map<String, dynamic>> queryComplete(QueryRequest request) async {
    try {
      final user = FirebaseAuth.instance.currentUser;

      if (user == null) {
        throw Exception('User not logged in');
      }

      print('🔍 Query complete request (non-streaming)...');
      print('👤 User ID: ${user.uid}');
      print('📝 Question: ${request.question}');

      final response = await http.post(
        Uri.parse('$baseUrl/query'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-User-ID': user.uid,
        },
        body: json.encode(request.toJson()),
      ).timeout(const Duration(seconds: 30));

      print('📥 Response status: ${response.statusCode}');
      print('📥 Response body: ${response.body}');

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        final errorBody = response.body;
        try {
          final errorJson = json.decode(errorBody);
          throw Exception(errorJson['detail'] ?? errorBody);
        } catch (_) {
          throw Exception('Query failed: $errorBody');
        }
      }
    } catch (e) {
      print('❌ Query error: $e');
      rethrow;
    }
  }
}