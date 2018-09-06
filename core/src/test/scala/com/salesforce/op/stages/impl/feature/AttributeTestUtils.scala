package com.salesforce.op.stages.impl.feature

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.sql.types.StructField
import org.junit.runner.RunWith
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
object AttributeTestUtils extends Matchers{

  final def assertNominal(schema: StructField, expectedNominal: Array[Boolean]) = {
    val attributes = AttributeGroup.fromStructField(schema).attributes.get
    attributes.map(_.isNominal) shouldBe expectedNominal
  }
}
