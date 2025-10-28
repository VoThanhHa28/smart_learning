import 'package:flutter/foundation.dart';
import 'dart:io' show Platform;

class ApiConfig {
  // ✅ Cấu hình IP (thay đổi nếu cần)
  static const String _localIp = '192.168.1.100'; // ← Đổi IP này nếu cần

  /// Lấy base URL dựa trên platform
  static String get baseUrl {
    // 1. Nếu chạy trên Web
    if (kIsWeb) {
      return 'http://localhost:8000';
    }

    // 2. Nếu chạy trên mobile/desktop
    try {
      if (Platform.isAndroid) {
        // Android Emulator: 10.0.2.2 trỏ tới localhost của máy host
        return 'http://10.0.2.2:8000';

        // ⚠️ Nếu chạy trên thiết bị Android thật, uncomment dòng dưới:
        // return 'http://$_localIp:8000';
      } else if (Platform.isIOS) {
        // iOS Simulator: dùng localhost OK
        return 'http://localhost:8000';

        // ⚠️ Nếu chạy trên iPhone thật, uncomment dòng dưới:
        // return 'http://$_localIp:8000';
      } else if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
        // Desktop platforms
        return 'http://localhost:8000';
      }
    } catch (e) {
      // Fallback nếu có lỗi
      print('⚠️ Error detecting platform: $e');
      return 'http://localhost:8000';
    }

    // Default fallback
    return 'http://localhost:8000';
  }

  /// Timeout cho HTTP requests
  static const Duration requestTimeout = Duration(seconds: 30);

  /// Timeout cho upload files
  static const Duration uploadTimeout = Duration(seconds: 120);

  /// Timeout cho streaming query
  static const Duration streamTimeout = Duration(seconds: 60);

  /// In thông tin debug khi app khởi động
  static void printDebugInfo() {
    print('=' * 60);
    print('🌐 API Configuration');
    print('=' * 60);
    print('  Base URL: $baseUrl');
    print('  Platform: ${_getPlatformName()}');
    print('  Is Web: $kIsWeb');
    print('  Request Timeout: ${requestTimeout.inSeconds}s');
    print('  Upload Timeout: ${uploadTimeout.inSeconds}s');
    print('=' * 60);
  }

  static String _getPlatformName() {
    if (kIsWeb) return 'Web';
    try {
      if (Platform.isAndroid) return 'Android';
      if (Platform.isIOS) return 'iOS';
      if (Platform.isMacOS) return 'macOS';
      if (Platform.isWindows) return 'Windows';
      if (Platform.isLinux) return 'Linux';
    } catch (e) {
      return 'Unknown';
    }
    return 'Unknown';
  }
}