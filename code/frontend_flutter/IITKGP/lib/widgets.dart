import 'package:flutter/material.dart';
import 'main.dart';
import 'task1.dart';

class MainView extends StatefulWidget {


  const MainView({
    Key? key,
  }) : super(key: key);

  @override
  _MainViewState createState() => _MainViewState();
}

class _MainViewState extends State<MainView> {
  int _selectedIndex = 0;

  late final List<Widget> _screens;

  @override
  void initState() {
    super.initState();
    _screens = [
      MyHomePage(),
      MyHomePage(),
    ];
  }

  void _onTabTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onTabTapped,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.black,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.task_alt),
            label: 'Task1',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.task_alt),
            label: 'Task2',
          ),

        ],
      ),
    );
  }
}
