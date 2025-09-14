import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:smart_learning/features/auth/presentation/auth_gate.dart';
import 'package:smart_learning/features/library/presentation/library_screen.dart';


final router = GoRouter(
  routes: [
    GoRoute(path: '/', builder: (_, __) => const AuthGate()),
    GoRoute(path: '/library', builder: (_, __) => const LibraryScreen()),
  ],
);

class App extends StatelessWidget {
  const App({super.key});
  @override
  Widget build(BuildContext context) {
    return ProviderScope(
      child: MaterialApp.router(
        debugShowCheckedModeBanner: false,
        theme: ThemeData(useMaterial3: true),
        routerConfig: router,
      ),
    );
  }
}
