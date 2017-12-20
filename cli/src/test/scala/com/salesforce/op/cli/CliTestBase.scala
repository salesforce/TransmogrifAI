/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import java.io.{ByteArrayOutputStream, File, FileWriter, StringReader}

import com.salesforce.op.test.TestCommon
import org.scalatest.{Assertions, BeforeAndAfter, FlatSpec}

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

/**
 * Test for generator operations
 */
class CliTestBase extends FlatSpec with TestCommon with Assertions with BeforeAndAfter {
  CommandParser.AUTO_ENABLED = true

  protected val ProjectName = "CliGeneratedTestProject"

  trait Outcome

  case class Crashed(msg: String, code: Int) extends Throwable with Outcome {
    override def toString: String = s"Crashed($msg, $code)"
  }

  case object Succeeded extends Outcome

  case class Result(outcome: Outcome, out: String, err: String)


  class Sut(input: String = "") extends CliExec {
    override protected val DEBUG = true
    override def quit(errorMsg: String, code: Int = -1): Nothing = {
      throw Crashed(errorMsg, code)
    }

    def run(args: String*): Result = {
      val out = new ByteArrayOutputStream
      val err = new ByteArrayOutputStream

      val outcome =
        Console.withIn(new StringReader(input)) {
          Console.withOut(out) {
            Console.withErr(err) {
              try {
                super.main(args.toArray)
                Succeeded
              } catch {
                case oops: Crashed =>
                  Console.err.println(oops.msg)
                  oops
              }
            }
          }
        }
      Result(outcome, out.toString, err.toString)
    }
  }

  before {
    new Sut().delete(new File(ProjectName))
  }

  after {
    new Sut().delete(new File(ProjectName))
  }

  val expectedSourceFiles = "Features.scala" :: s"$ProjectName.scala"::Nil

  val projectDir = new File(ProjectName.toLowerCase)

  def checkAvroFile(source: File): Unit = {
    val avroFile = new File(projectDir, s"src/main/avro/${source.getName}")
    avroFile should exist
    Source.fromFile(avroFile).getLines.mkString("\n") shouldBe
      Source.fromFile(source).getLines.mkString("\n")
  }

  def checkScalaFiles(shouldNotContain: String): Unit = {
    val srcDir = new File(projectDir, "src/main/scala/com/salesforce/app")
    srcDir should exist

    for {
      filename <- expectedSourceFiles
      file = new File(srcDir, filename)
      _ = file should exist
      contents = Source.fromFile(file).getLines.zipWithIndex
      (line, n) <- contents
    } {
      withClue(s"File $file, line $n\n$line") {
        line.contains(shouldNotContain) shouldBe false
      }
    }
  }

  protected val TestAvsc = "test-data/PassengerDataAll.avsc"
  protected val TestCsvHeadless = "test-data/PassengerDataAll.csv"
  protected val TestCsvWithHeaders = "test-data/PassengerDataAllWithHeader.csv"
  protected lazy val dataPath = new File(".").getAbsolutePath + "/" + TestCsvHeadless

  protected def appRuntimeArgs(whatToDo: String) =
    s"--run-type=$whatToDo --model-location=/tmp/titanic-model " +
    s"--read-location Passenger=$dataPath"
}
