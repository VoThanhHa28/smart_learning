import 'dart:io';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:file_picker/file_picker.dart';

class LibraryScreen extends StatefulWidget {
  const LibraryScreen({super.key});
  @override
  State<LibraryScreen> createState() => _LibraryScreenState();
}

class _LibraryScreenState extends State<LibraryScreen> {
  final user = FirebaseAuth.instance.currentUser!;

  Future<void> _pickAndUpload() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );
    if (result == null) return;

    final file = result.files.single;
    final filePath = file.path!;
    final fileName = file.name;

    final storagePath = 'documents/${user.uid}/$fileName';
    final ref = FirebaseStorage.instance.ref(storagePath);

    // Upload
    await ref.putFile(File(filePath));

    // Lưu metadata Firestore
    await FirebaseFirestore.instance.collection('documents').add({
      'userId': user.uid,
      'fileName': fileName,
      'storagePath': storagePath,
      'uploadedAt': FieldValue.serverTimestamp(),
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade50,
      appBar: AppBar(
        title: const Text('📚 Your Library'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async => FirebaseAuth.instance.signOut(),
          ),
        ],
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance
            .collection('documents')
            .where('userId', isEqualTo: user.uid)
            .orderBy('uploadedAt', descending: true)
            .snapshots(),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return const Center(child: Text('⚠️ Error loading documents'));
          }
          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final docs = snapshot.data!.docs;
          if (docs.isEmpty) {
            return const Center(
              child: Text('No documents yet.\nTap the button to upload 📄',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.grey)),
            );
          }
          return ListView.separated(
            padding: const EdgeInsets.all(16),
            itemCount: docs.length,
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemBuilder: (context, index) {
              final data = docs[index].data()! as Map<String, dynamic>;
              final fileName = data['fileName'] ?? 'Untitled';
              final uploadedAt = data['uploadedAt']?.toDate().toString() ?? '';
              return Card(
                elevation: 2,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                child: ListTile(
                  leading: const Icon(Icons.picture_as_pdf,
                      size: 36, color: Colors.red),
                  title: Text(fileName,
                      style: const TextStyle(
                          fontWeight: FontWeight.bold, fontSize: 16)),
                  subtitle: Text(uploadedAt,
                      style: const TextStyle(color: Colors.grey)),
                  trailing: IconButton(
                    icon: const Icon(Icons.more_vert),
                    onPressed: () {
                      // TODO: hiển thị menu (delete, share, index...)
                    },
                  ),
                  onTap: () {
                    // TODO: mở PDF hoặc gửi sang backend
                  },
                ),
              );
            },
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _pickAndUpload,
        icon: const Icon(Icons.upload_file),
        label: const Text('Upload PDF'),
      ),
    );
  }
}
