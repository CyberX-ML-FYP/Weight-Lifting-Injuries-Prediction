import { Outlet } from "react-router-dom";
import AppNavbar from "./AppNavbar";
import AppFooter from "./AppFooter";

export default function Layout() {
  return (
    <div className="app-shell d-flex flex-column min-vh-100">
      <AppNavbar />
      <main className="app-main flex-grow-1">
        <Outlet />
      </main>
      <AppFooter />
    </div>
  );
}
