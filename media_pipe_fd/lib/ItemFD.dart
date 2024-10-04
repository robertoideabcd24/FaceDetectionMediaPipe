// lib/face_detection_page.dart

import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

class FaceDetectionPage extends StatefulWidget {
  @override
  _FaceDetectionPageState createState() => _FaceDetectionPageState();
}

class _FaceDetectionPageState extends State<FaceDetectionPage> {
  late CameraController _cameraController;
  late Interpreter _interpreter;
  bool _isDetecting = false;
  List<Rect> _faces = [];

  // Tamaño de entrada requerido por BlazeFace
  final int inputSize = 128; // BlazeFace suele usar 128x128

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _interpreter.close();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    // Obtener la lista de cámaras disponibles
    final cameras = await availableCameras();
    final camera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first);

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController.initialize();
    _cameraController.startImageStream(_processCameraImage);

    setState(() {});
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('models/blazeface.tflite');
      print('Modelo BlazeFace cargado exitosamente');
    } catch (e) {
      print('Error al cargar el modelo: $e');
    }
  }

  void _processCameraImage(CameraImage image) {
    if (_isDetecting) return;

    _isDetecting = true;

    // Convertir CameraImage a formato que BlazeFace pueda procesar
    _convertCameraImage(image).then((convertedImage) {
      if (convertedImage == null) {
        _isDetecting = false;
        return;
      }

      // Redimensionar la imagen al tamaño de entrada del modelo
      img.Image resizedImage =
          img.copyResize(convertedImage, width: inputSize, height: inputSize);

      // Preprocesamiento: Normalizar los valores de píxeles
      TensorImage inputImage = TensorImage.fromImage(resizedImage);
      ImageProcessor imageProcessor = ImageProcessorBuilder()
          .add(ResizeOp(inputSize, inputSize, ResizeMethod.BILINEAR))
          .add(NormalizeOp(0, 1)) // Normalizar entre 0 y 1
          .build();
      inputImage = imageProcessor.process(inputImage);

      // Ejecutar la inferencia
      TensorBuffer outputLocations =
          TensorBuffer.createFixedSize([896, 4], TfLiteType.float32);
      TensorBuffer outputScores =
          TensorBuffer.createFixedSize([896], TfLiteType.float32);

      Map<int, Object> outputs = {
        0: outputLocations.buffer,
        1: outputScores.buffer,
      };

      _interpreter.runForMultipleInputs([inputImage.buffer], outputs);

      // Procesar los resultados
      List<Rect> detectedFaces = _processOutput(
        outputLocations.getDoubleList(),
        outputScores.getDoubleList(),
        image.width,
        image.height,
      );

      setState(() {
        _faces = detectedFaces;
      });

      _isDetecting = false;
    });
  }

  Future<img.Image?> _convertCameraImage(CameraImage image) async {
    try {
      // Convertir YUV420 a RGB
      final img.Image convertedImage = _yuv420ToImage(image);
      return convertedImage;
    } catch (e) {
      print('Error al convertir CameraImage: $e');
      return null;
    }
  }

  img.Image _yuv420ToImage(CameraImage image) {
    // Convertir YUV420 a RGB usando la biblioteca 'image'
    final int width = image.width;
    final int height = image.height;
    final img.Image imgImage = img.Image(width, height);

    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final Uint8List bytesY = planeY.bytes;
    final Uint8List bytesU = planeU.bytes;
    final Uint8List bytesV = planeV.bytes;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = (y ~/ 2) * (width ~/ 2) + (x ~/ 2);
        final int yp = bytesY[y * width + x];
        final int up = bytesU[uvIndex];
        final int vp = bytesV[uvIndex];

        double r = yp + 1.370705 * (vp - 128);
        double g = yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128);
        double b = yp + 1.732446 * (up - 128);

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        imgImage.setPixelRgba(x, y, r.toInt(), g.toInt(), b.toInt());
      }
    }

    return imgImage;
  }

  List<Rect> _processOutput(List<double> locations, List<double> scores,
      int originalWidth, int originalHeight) {
    List<Rect> faces = [];

    // Umbral de confianza
    double threshold = 0.5;

    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > threshold) {
        double yMin = locations[i * 4];
        double xMin = locations[i * 4 + 1];
        double yMax = locations[i * 4 + 2];
        double xMax = locations[i * 4 + 3];

        // Convertir coordenadas al tamaño original de la imagen
        double left = xMin * originalWidth;
        double top = yMin * originalHeight;
        double right = xMax * originalWidth;
        double bottom = yMax * originalHeight;

        faces.add(Rect.fromLTRB(left, top, right, bottom));
      }
    }

    return faces;
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: Text('Detección de Rostros')),
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text('Detección de Rostros con BlazeFace')),
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          CustomPaint(
            painter: FacePainter(_faces, _cameraController.value.previewSize!),
          ),
        ],
      ),
    );
  }
}

class FacePainter extends CustomPainter {
  final List<Rect> faces;
  final Size imageSize;

  FacePainter(this.faces, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / imageSize.height;
    final double scaleY = size.height / imageSize.width;

    final Paint paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    for (Rect face in faces) {
      // Ajustar las coordenadas según la orientación de la cámara
      Rect transformedRect = Rect.fromLTRB(
        face.left * scaleX,
        face.top * scaleY,
        face.right * scaleX,
        face.bottom * scaleY,
      );
      canvas.drawRect(transformedRect, paint);
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faces != faces;
  }
}
