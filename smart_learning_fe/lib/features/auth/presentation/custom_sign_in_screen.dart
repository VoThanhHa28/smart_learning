import 'package:flutter/material.dart';
import 'package:firebase_ui_auth/firebase_ui_auth.dart' as fui;

class CustomSignInScreen extends StatelessWidget {
  const CustomSignInScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Đảm bảo màn hình có thể cuộn khi bị tràn
      body: SafeArea(
        child: SingleChildScrollView(
          child: SizedBox(
            height: MediaQuery.of(context).size.height -
                MediaQuery.of(context).padding.top -
                MediaQuery.of(context).padding.bottom,
            child: fui.SignInScreen(
              providers: [fui.EmailAuthProvider()],
              headerBuilder: (context, constraints, _) => Padding(
                padding: const EdgeInsets.all(10.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.school_rounded,
                      size: 32,
                      color: Theme.of(context).primaryColor,
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Smart Learning',
                      style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).primaryColor,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Your personal learning assistant',
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
              styles: const {
                fui.EmailFormStyle(
                  signInButtonVariant: fui.ButtonVariant.filled,
                ),
              },
              subtitleBuilder: (context, action) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8.0),
                  child: action == fui.AuthAction.signIn
                      ? const Text('Welcome back! Please sign in to continue.')
                      : const Text('Create an account to get started!'),
                );
              },
              footerBuilder: (context, _) {
                return Padding(
                  padding: const EdgeInsets.only(top: 16.0, bottom: 16.0),
                  child: Text(
                    '© 2025 Smart Learning',
                    style: Theme.of(context).textTheme.bodySmall,
                    textAlign: TextAlign.center,
                  ),
                );
              },
              showAuthActionSwitch: true,
            ),
          ),
        ),
      ),
    );
  }
}