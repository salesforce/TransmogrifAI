package com.salesforce.op.stages

import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.Matchers._

@RunWith(classOf[JUnitRunner])
class ColumnMetadataParamTest extends FlatSpec {

  val p1 = new ColumnMetadataParam("", "p1", "p1 doc")
  val columnMetadata = ColumnMetadata.fromElems(
    "col1" -> Metadata.empty,
    "col2" -> Metadata.fromJson("""{"attr": "feature1"}""")
  )
  val json = """{"col1":{},"col2":{"attr":"feature1"}}"""

  it should "serialize to json" in {
    p1.jsonEncode(columnMetadata) shouldBe json
  }

  it should "deserialize from json" in {
    p1.jsonDecode(json) shouldBe columnMetadata
  }

  it should "complete a json serialization/deserialization round-trip" in {
    p1.jsonDecode(p1.jsonEncode(columnMetadata)) shouldBe columnMetadata
  }
}
