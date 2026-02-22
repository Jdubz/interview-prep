import { useState } from 'react';
import CustomHooksDemo from './lessons/02-custom-hooks/CustomHooksDemo';
import './App.css';

const lessons = [
  { id: '01', title: 'Hooks Deep Dive', component: null },
  { id: '02', title: 'Custom Hooks', component: CustomHooksDemo },
  { id: '03', title: 'Performance Optimization', component: null },
  { id: '04', title: 'Advanced Patterns', component: null },
  { id: '05', title: 'State Management', component: null },
  { id: '06', title: 'RSC & Suspense', component: null },
  { id: '07', title: 'Testing React', component: null },
  { id: '08', title: 'React Internals', component: null },
  { id: '09', title: 'Event Loop & Async', component: null },
  { id: '10', title: 'Streams & Buffers', component: null },
  { id: '11', title: 'Error Handling', component: null },
  { id: '12', title: 'Auth & Security', component: null },
  { id: '13', title: 'Database Patterns', component: null },
  { id: '14', title: 'API Design', component: null },
  { id: '15', title: 'Testing Node', component: null },
  { id: '16', title: 'Performance & Scaling', component: null },
];

function App() {
  const [activeLesson, setActiveLesson] = useState('02');

  const CurrentLesson = lessons.find(l => l.id === activeLesson)?.component;

  return (
    <div className="app">
      <nav className="sidebar">
        <h2>Interview Prep</h2>
        <div className="section-label">React</div>
        {lessons.slice(0, 8).map(lesson => (
          <button
            key={lesson.id}
            className={`nav-item ${activeLesson === lesson.id ? 'active' : ''} ${!lesson.component ? 'disabled' : ''}`}
            onClick={() => lesson.component && setActiveLesson(lesson.id)}
          >
            {lesson.id}. {lesson.title}
          </button>
        ))}
        <div className="section-label">Node.js</div>
        {lessons.slice(8).map(lesson => (
          <button
            key={lesson.id}
            className={`nav-item ${activeLesson === lesson.id ? 'active' : ''} ${!lesson.component ? 'disabled' : ''}`}
            onClick={() => lesson.component && setActiveLesson(lesson.id)}
          >
            {lesson.id}. {lesson.title}
          </button>
        ))}
      </nav>

      <main className="content">
        {CurrentLesson ? (
          <CurrentLesson />
        ) : (
          <div className="placeholder">
            <h1>Lesson {activeLesson}</h1>
            <p>This lesson is coming up next in our tutoring session.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
