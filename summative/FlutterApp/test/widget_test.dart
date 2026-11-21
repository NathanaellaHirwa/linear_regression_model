// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:garment_productivity_predictor/main.dart';

void main() {
  testWidgets('one labelled text field per model input, a Predict button and a display area', (tester) async {
    await tester.pumpWidget(const GarmentProductivityApp());

    expect(find.byType(TextField), findsNWidgets(kFields.length));
    expect(kFields.length, 12);
    expect(find.text('Predict'), findsOneWidget);
    expect(find.text('Result'), findsOneWidget);
  });

  testWidgets('empty form shows readable errors in the display area', (tester) async {
    await tester.pumpWidget(const GarmentProductivityApp());

    await tester.ensureVisible(find.text('Predict'));
    await tester.tap(find.text('Predict'));
    await tester.pumpAndSettle();

    expect(find.text('Please check your input'), findsOneWidget);
    expect(find.textContaining('Team is required'), findsOneWidget);
  });

  testWidgets('wrong type and out-of-range values are reported', (tester) async {
    await tester.pumpWidget(const GarmentProductivityApp());

    // team is the 3rd field, target productivity the 5th
    final fields = find.byType(TextField);
    await tester.enterText(fields.at(2), 'abc');
    await tester.enterText(fields.at(4), '1.7');
    await tester.ensureVisible(find.text('Predict'));
    await tester.tap(find.text('Predict'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Team must be a whole number'), findsOneWidget);
    expect(find.textContaining('Target productivity must be between 0.05 and 1.0'), findsOneWidget);
  });
}
