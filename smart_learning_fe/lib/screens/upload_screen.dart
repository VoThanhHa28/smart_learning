import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:file_picker/file_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:smart_learning/screens/query_screen.dart';
import '../../../services/api_service.dart';
import '../../../models/upload_response.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> with SingleTickerProviderStateMixin {
  final ApiService _apiService = ApiService();
  final TextEditingController _subjectController = TextEditingController();
  final TextEditingController _courseIdController = TextEditingController();

  PlatformFile? _selectedPlatformFile;
  bool _isUploading = false;
  double _uploadProgress = 0.0;
  UploadResponse? _uploadResponse;
  User? _currentUser;
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _currentUser = FirebaseAuth.instance.currentUser;
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );
  }

  @override
  void dispose() {
    _subjectController.dispose();
    _courseIdController.dispose();
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _pickPdfFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        withData: true,
      );

      if (result != null && result.files.isNotEmpty) {
        setState(() {
          _selectedPlatformFile = result.files.first;
        });
        print('✅ File selected: ${_selectedPlatformFile!.name}');
        _showSuccess('Đã chọn file: ${_selectedPlatformFile!.name}');
      }
    } catch (e) {
      print('❌ File picker error: $e');
      _showError('Lỗi chọn file: $e');
    }
  }

  Future<void> _uploadPdf() async {
    if (_selectedPlatformFile == null) {
      _showError('Vui lòng chọn file PDF');
      return;
    }

    if (_subjectController.text.trim().isEmpty) {
      _showError('Vui lòng nhập môn học');
      return;
    }

    if (_currentUser == null) {
      _showError('Bạn chưa đăng nhập');
      return;
    }

    setState(() {
      _isUploading = true;
      _uploadProgress = 0.0;
      _uploadResponse = null;
    });

    _animationController.repeat();

    try {
      print('📤 Starting upload...');
      print('👤 User: ${_currentUser!.email}');
      print('📝 Subject: ${_subjectController.text.trim()}');

      // Simulate progress (since actual progress tracking is complex)
      _simulateProgress();

      final response = await _apiService.uploadPdf(
        platformFile: _selectedPlatformFile!,
        subject: _subjectController.text.trim(),
        userId: _currentUser!.uid,
        courseId: _courseIdController.text.trim().isNotEmpty
            ? _courseIdController.text.trim()
            : null,
      );

      print('✅ Upload completed');

      if (!mounted) return;

      setState(() {
        _uploadResponse = response;
        _uploadProgress = 1.0;
        _isUploading = false;
      });

      _animationController.stop();

      final chunks = response.chunks ?? 0;

      if (chunks > 0) {
        _showSuccess('Upload thành công! $chunks chunks đã được lưu.');

        await Future.delayed(const Duration(milliseconds: 1500));

        if (!mounted) return;

        // Navigate to QueryScreen
        await Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => QueryScreen(
              subject: _subjectController.text.trim(),
              courseId: response.courseId,
              docId: response.docId,
            ),
          ),
        );
      } else {
        _showWarning('Upload thành công nhưng không có chunks');
      }
    } catch (e, stackTrace) {
      print('❌ Upload failed: $e');
      print('Stack: $stackTrace');

      if (!mounted) return;

      setState(() {
        _isUploading = false;
        _uploadProgress = 0.0;
      });

      _animationController.stop();
      _showError('Lỗi upload: $e');
    }
  }

  void _simulateProgress() {
    // Simulate upload progress
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_isUploading && mounted) {
        setState(() {
          _uploadProgress = (_uploadProgress + 0.05).clamp(0.0, 0.95);
        });
        if (_uploadProgress < 0.95) {
          _simulateProgress();
        }
      }
    });
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.white),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.red.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  void _showSuccess(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.check_circle_outline, color: Colors.white),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.green.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  void _showWarning(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.warning_outlined, color: Colors.white),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.orange.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async {
        if (_isUploading) {
          _showWarning('Đang upload, vui lòng đợi...');
          return false;
        }
        return true;
      },
      child: Scaffold(
        backgroundColor: Colors.grey[50],
        appBar: AppBar(
          elevation: 0,
          backgroundColor: Colors.white,
          foregroundColor: Colors.black87,
          leading: _isUploading
              ? null
              : IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () {
              if (mounted && Navigator.canPop(context)) {
                Navigator.pop(context);
              }
            },
          ),
          title: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Upload Tài Liệu',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (_currentUser != null)
                Text(
                  _currentUser!.email ?? 'No email',
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey[600],
                    fontWeight: FontWeight.normal,
                  ),
                ),
            ],
          ),
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(1),
            child: Container(
              height: 1,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.blue.shade200,
                    Colors.purple.shade200,
                  ],
                ),
              ),
            ),
          ),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header icon with animation
              if (!_isUploading)
                Container(
                  padding: const EdgeInsets.all(32),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.blue.shade100, Colors.purple.shade100],
                    ),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.cloud_upload,
                    size: 80,
                    color: Colors.blue.shade700,
                  ),
                ),

              if (_isUploading)
                RotationTransition(
                  turns: _animationController,
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [Colors.blue.shade100, Colors.purple.shade100],
                      ),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      Icons.upload_file,
                      size: 80,
                      color: Colors.blue.shade700,
                    ),
                  ),
                ),

              const SizedBox(height: 24),

              Text(
                _isUploading ? 'Đang xử lý...' : 'Upload tài liệu PDF',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey[800],
                ),
                textAlign: TextAlign.center,
              ),

              if (_isUploading) ...[
                const SizedBox(height: 8),
                Text(
                  'Vui lòng không tắt ứng dụng',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[600],
                  ),
                  textAlign: TextAlign.center,
                ),
              ],

              const SizedBox(height: 32),

              // Progress indicator
              if (_isUploading) ...[
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.05),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      LinearProgressIndicator(
                        value: _uploadProgress,
                        backgroundColor: Colors.grey[200],
                        valueColor: AlwaysStoppedAnimation<Color>(
                          Colors.blue.shade600,
                        ),
                        minHeight: 8,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        '${(_uploadProgress * 100).toInt()}%',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[800],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
              ],

              // Subject field
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.03),
                      blurRadius: 10,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: TextField(
                  controller: _subjectController,
                  decoration: InputDecoration(
                    labelText: 'Môn học *',
                    hintText: 'Ví dụ: Toán, Vật Lý, Hóa Học',
                    prefixIcon: Icon(Icons.subject, color: Colors.blue.shade600),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide.none,
                    ),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                  enabled: !_isUploading,
                ),
              ),

              const SizedBox(height: 16),

              // Course ID field (optional)
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.03),
                      blurRadius: 10,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: TextField(
                  controller: _courseIdController,
                  decoration: InputDecoration(
                    labelText: 'Course ID (tùy chọn)',
                    hintText: 'Để trống để tự động tạo',
                    prefixIcon: Icon(Icons.tag, color: Colors.purple.shade600),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide.none,
                    ),
                    filled: true,
                    fillColor: Colors.white,
                  ),
                  enabled: !_isUploading,
                ),
              ),

              const SizedBox(height: 24),

              // File picker
              if (_selectedPlatformFile == null)
                InkWell(
                  onTap: _isUploading ? null : _pickPdfFile,
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: Colors.blue.shade200,
                        width: 2,
                        style: BorderStyle.solid,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.03),
                          blurRadius: 10,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        Icon(
                          Icons.upload_file,
                          size: 64,
                          color: Colors.blue.shade400,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Chọn file PDF',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.grey[800],
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Nhấn để chọn file từ thiết bị',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

              // Selected file display
              if (_selectedPlatformFile != null)
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.green.shade50, Colors.green.shade100],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: Colors.green.shade200,
                      width: 2,
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Icon(
                          Icons.picture_as_pdf,
                          color: Colors.red.shade400,
                          size: 32,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _selectedPlatformFile!.name,
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 15,
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              '${(_selectedPlatformFile!.size / 1024).toStringAsFixed(1)} KB',
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey[700],
                              ),
                            ),
                          ],
                        ),
                      ),
                      if (!_isUploading)
                        IconButton(
                          icon: const Icon(Icons.close, color: Colors.red),
                          onPressed: () {
                            setState(() {
                              _selectedPlatformFile = null;
                            });
                          },
                        ),
                    ],
                  ),
                ),

              const SizedBox(height: 32),

              // Upload button
              Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: _isUploading
                        ? [Colors.grey.shade400, Colors.grey.shade500]
                        : [Colors.blue.shade500, Colors.purple.shade500],
                  ),
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: _isUploading
                          ? Colors.grey.withOpacity(0.3)
                          : Colors.blue.withOpacity(0.3),
                      blurRadius: 8,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: ElevatedButton(
                  onPressed: _isUploading ? null : _uploadPdf,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    padding: const EdgeInsets.all(16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: _isUploading
                      ? const Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor:
                          AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      ),
                      SizedBox(width: 12),
                      Text(
                        'Đang upload...',
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  )
                      : const Text(
                    'Bắt đầu Upload',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),

              // Success response
              if (_uploadResponse != null) ...[
                const SizedBox(height: 32),
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.green.shade50, Colors.green.shade100],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: Colors.green.shade300,
                      width: 2,
                    ),
                  ),
                  child: Column(
                    children: [
                      Icon(
                        Icons.check_circle,
                        color: Colors.green.shade600,
                        size: 64,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Upload thành công!',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.green.shade700,
                        ),
                      ),
                      const Divider(height: 32),
                      _buildInfoRow(
                        'Course ID',
                        _uploadResponse!.courseId,
                        Icons.tag,
                      ),
                      _buildInfoRow(
                        'Số chunks',
                        _uploadResponse!.chunks?.toString() ?? '0',
                        Icons.analytics,
                      ),
                      if (_uploadResponse!.warnings.isNotEmpty) ...[
                        const SizedBox(height: 16),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.orange.shade50,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                Icons.warning_amber,
                                color: Colors.orange.shade600,
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  _uploadResponse!.warnings.join(", "),
                                  style: TextStyle(
                                    color: Colors.orange.shade800,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value, IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, size: 20, color: Colors.grey[600]),
          const SizedBox(width: 12),
          Text(
            '$label: ',
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 15,
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                fontSize: 15,
                color: Colors.grey[700],
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}