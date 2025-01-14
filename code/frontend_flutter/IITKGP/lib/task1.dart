import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool _isLoading = false;
  List<String> _results = [];

  // Function to pick PDF files
  Future<void> _pickFiles() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
      allowMultiple: true,
    );

    if (result != null) {
      List<File> files = result.paths.map((path) => File(path!)).toList();

      // Call the API to upload the files and get predictions
      _uploadFiles(files);
    }
  }

  // Function to upload files to Flask API
  Future<void> _uploadFiles(List<File> files) async {
    setState(() {
      _isLoading = true;
      _results = []; // Clear previous results
    });

    try {
      var uri = Uri.parse('http://10.0.2.2:5000/predict'); // For Android Emulator

      for (File file in files) {
        var request = http.MultipartRequest('POST', uri);
        request.files.add(await http.MultipartFile.fromPath('files', file.path));

        // Send the request and await the response
        var response = await request.send();

        if (response.statusCode == 200) {
          final responseData = await response.stream.bytesToString();
          try {
            final List<dynamic> jsonResponse = json.decode(responseData);
            // Parse and add results to the list
            for (var item in jsonResponse) {
              _results.add('${item['file_name']} is ${item['publishability_label']}');
            }
          } catch (e) {
            _results.add('Error parsing response for file: ${file.path}');
            print('JSON Parsing Error: $e');
          }
        } else {
          _results.add('Error uploading file: ${file.path}, Status Code: ${response.statusCode}');
        }
      }
    } catch (e) {
      _results.add('An error occurred: $e');
      print('Error: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Data science IITkGP'),
        centerTitle: true,
        backgroundColor: Colors.blueAccent,
      ),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Card(
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15.0),
                  ),
                  elevation: 5,
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      children: [
                        Text(
                          'Upload and Predict',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 10),
                        Text(
                          'Select PDF files to upload and receive prediction results.',
                          style: TextStyle(fontSize: 16, color: Colors.grey),
                          textAlign: TextAlign.center,
                        ),
                        SizedBox(height: 20),
                        ElevatedButton.icon(
                          onPressed: _pickFiles,
                          icon: Icon(Icons.file_upload),
                          label: Text('Pick PDF Files'),
                          style: ElevatedButton.styleFrom(
                            foregroundColor: Colors.white, backgroundColor: Colors.blueAccent,
                            padding: EdgeInsets.symmetric(
                              vertical: 15,
                              horizontal: 25,
                            ),
                            textStyle: TextStyle(fontSize: 18),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                SizedBox(height: 30),
                if (_results.isNotEmpty)
                  Expanded(
                    child: Card(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15.0),
                      ),
                      elevation: 5,
                      color: Colors.green[50],
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: ListView.builder(
                          itemCount: _results.length,
                          itemBuilder: (context, index) {
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 5.0),
                              child: Text(
                                _results[index],
                                style: TextStyle(fontSize: 16),
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: Center(
                child: CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
