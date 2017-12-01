/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.test.TestCommon
import org.apache.spark.sql.types.{MetadataBuilder, StructField}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class RichMetadataTest extends FlatSpec with TestCommon {

  import com.salesforce.op.utils.spark.RichMetadata._

  Spec[RichMetadata] should "create a metadata from a map" in {
    val expected = Map(
      "1" -> 1L, "2" -> 1.0, "3" -> true, "4" -> "1",
      "5" -> Array(1L), "6" -> Array(1.0), "6" -> Array(true), "7" -> Array("1")
    )
    val meta = expected.toMetadata
    meta.underlyingMap.toSeq shouldBe expected.toSeq
  }

  it should "throw an error on unsupported type in a map" in {
    the[RuntimeException] thrownBy Map("a" -> Map("b" -> 1)).toMetadata
  }

}
