import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ThemeProvider } from "next-themes";
import ThemeToggle from "../components/ThemeToggle";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Leapfrog - Frog Classification Tool",
  description: "A tool for tracking and classifying frogs from camera imagery",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} apple-gradient min-h-screen`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <ThemeToggle />
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
