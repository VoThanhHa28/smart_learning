import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:go_router/go_router.dart';
import 'package:firebase_auth/firebase_auth.dart' as fa show FirebaseAuth;
import 'package:smart_learning/features/auth/presentation/custom_sign_in_screen.dart';

class AuthGate extends StatelessWidget {
  const AuthGate({super.key});
  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: fa.FirebaseAuth.instance.authStateChanges(),
      builder: (context, snap) {
        if (snap.connectionState == ConnectionState.waiting) {
          return const Scaffold(body: Center(child: CircularProgressIndicator()));
        }
        if (snap.hasData) {
          Future.microtask(() => context.go('/library'));
          return const SizedBox.shrink();
        }

        // Sử dụng màn hình đăng nhập tùy chỉnh
        return const CustomSignInScreen();
      },
    );
  }
}