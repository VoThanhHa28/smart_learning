import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:smart_learning/features/auth/presentation/login_screen.dart';
import 'package:smart_learning/screens/home_screen.dart';
import 'package:smart_learning/screens/upload_screen.dart';

class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        // Đang kiểm tra trạng thái auth
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(
              child: CircularProgressIndicator(),
            ),
          );
        }

        // Nếu có lỗi
        if (snapshot.hasError) {
          return Scaffold(
            body: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: Colors.red,
                    size: 60,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Lỗi xác thực: ${snapshot.error}',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.red),
                  ),
                ],
              ),
            ),
          );
        }

        // Đã đăng nhập -> chuyển đến UploadScreen
        if (snapshot.hasData && snapshot.data != null) {
          return const HomeScreen();
        }

        // Chưa đăng nhập -> chuyển đến LoginScreen
        return const LoginScreen();
      },
    );
  }
}