/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import java.io.File

import org.apache.log4j.{Level, LogManager, Logger}
import org.scalatest._

import scala.collection.JavaConverters._
import scala.io.Source
import scala.reflect.{ClassTag, _}

/**
 * Trait with test commons such as Spec and Resource functions
 */
trait TestCommon extends Matchers with Assertions {

  /**
   * Set logging level for
   */
  def loggingLevel(level: Level): Unit = {
    val loggers = Logger.getRootLogger :: LogManager.getCurrentLoggers.asScala.toList
    loggers.collect { case l: Logger => l }.foreach(_.setLevel(level))
  }

  /**
   * Disable all logging
   */
  def loggingOff(): Unit = loggingLevel(Level.OFF)

  /**
   * Spec name for a class
   */
  case object Spec {
    def apply[T: ClassTag]: String = apply(classTag[T].runtimeClass)
    def apply[T1: ClassTag, T2: ClassTag]: String = apply[T2] + "[" + apply[T1] + "]"

    def apply(klazz: Class[_]): String = klazz.getSimpleName.stripSuffix("$")
  }

  /**
   * Load a test resource file
   *
   * @param parent resource folder
   * @param name   resource name
   * @return resource file
   */
  def resourceFile(parent: String = "src/test/resources", name: String): File = {
    val file = new File(parent, name)
    if (!file.canRead) throw new IllegalStateException(s"File $file unreadable")
    file
  }

  /**
   * Load a test resource file as string
   *
   * @param parent   resource folder
   * @param noSpaces trim all spaces
   * @param name     resource name
   * @return resource file
   */
  @deprecated("Use loadResource", "3.2.3")
  def resourceString(parent: String = "src/test/resources", noSpaces: Boolean = true, name: String): String = {
    val file = resourceFile(parent = parent, name = name)
    val contents = Source.fromFile(file, "UTF-8").mkString
    if (noSpaces) contents.replaceAll("\\s", "") else contents
  }

  /**
   * Loads resource by path
   * @param path absolute or relative path of a resource
   * @return the whole content of resource file as a string
   */
  def loadResource(path: String, noSpaces: Boolean = false): String = {
    val raw = Source.fromInputStream(getClass.getResourceAsStream(path)).getLines
    if (noSpaces) raw.mkString("").replaceAll("\\s", "") else raw.mkString("\n")
  }

}
