% Ackley Function GIF
[x, y] = meshgrid(-32:0.5:32, -32:0.5:32);
z = arrayfun(@(i,j) ackley([i j]), x, y);
animate_surface(x, y, z, 'Ackley Function', 'ackley.gif');

% Sphere Function (Modified)
[x, y] = meshgrid(-10:0.5:10, -10:0.5:10);
z = arrayfun(@(i,j) spherefmod([i j 0 0 0 0]), x, y);  % fill in dummy values
animate_surface(x, y, z, 'Sphere Function', 'sphere.gif');

[x, y] = meshgrid(-5.12:0.1:5.12, -5.12:0.1:5.12);
z = arrayfun(@(i,j) rastr([i j]), x, y);
animate_surface(x, y, z, 'Rastrigin Function', 'rastrigin.gif');

[x, y] = meshgrid(-1:0.05:1, -1:0.05:1);
z = arrayfun(@(i,j) rosensc([i j 0 0]), x, y);  % fill in 4D
animate_surface(x, y, z, 'Rosenbrock Function', 'rosenbrock.gif');

[x, y] = meshgrid(-600:10:600, -600:10:600);
z = arrayfun(@(i,j) griewank([i j]), x, y);
animate_surface(x, y, z, 'Griewank Function', 'griewank.gif');

[x, y] = meshgrid(-500:10:500, -500:10:500);
z = arrayfun(@(i,j) schwef([i j]), x, y);
animate_surface(x, y, z, 'Schwefel Function', 'schwefel.gif');

function animate_surface(x, y, z, title_str, filename)
    fig = figure('Visible', 'off');  % prevent UI popping up
    surf(x, y, z, 'EdgeColor', 'none');
    colormap('jet');
    xlabel('x₁'); ylabel('x₂'); zlabel('f(x)');
    title(title_str);
    axis tight;
    view(-45, 30);
    set(gcf,'color',[0 0 0]);

    for angle = 1:4:360
        view(angle, 30);
        drawnow
        frame = getframe(fig);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if angle == 1
            imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
        else
            imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end
    close(fig);
end
