import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_ui_auth/firebase_ui_auth.dart' as fui;
import 'package:go_router/go_router.dart';
import 'package:firebase_auth/firebase_auth.dart' as fa show FirebaseAuth;

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
        return fui.SignInScreen(
          providers: [fui.EmailAuthProvider()],
          headerBuilder: (context, _, __) => const Padding(
            padding: EdgeInsets.all(16),
            child: Text('Welcome to Smart Learning',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          ),
        );
      },
    );
  }
}
