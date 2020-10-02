package com.salesforce.op.utils.io.file

import com.salesforce.op.test.{TempDirectoryTest, TestCommon}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class FileCompressTest extends FlatSpec with TempDirectoryTest with TestCommon{

  Spec(FileCompress.getClass) should "compress a directory into a zip file" in {

  }

  it should "uncompress a zip file into a directory" in {

  }

  it should "round trip a file and get the same values" in {

  }

}
